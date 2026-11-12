import ast
from typing import Any, List, Tuple, cast

import polars as pl

from .ops import OP_CATEGORY, OPS_MAP


class ExpressionParser:
    """
    Parses string expressions into Polars Expressions.

    Example: "Rank(Ts_Mean(Close, 10))" -> pl.Expr.
    """

    def parse(self, expr_str: str) -> pl.Expr:
        """Parse a factor expression string."""
        # Clean up string
        expr_str = expr_str.strip()

        # Parse using Python's AST
        try:
            tree = ast.parse(expr_str, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {expr_str}") from e

        result = self._visit(tree.body)

        if isinstance(result, pl.Expr):
            return result
        if hasattr(result, "rolling"):
            return cast(pl.Expr, result)

        return pl.lit(result)

    def plan(self, expr_str: str) -> List[Tuple[str, str]]:
        """
        Generate an execution plan to handle nested window functions.

        This method detects conflicts between orthogonal partition types (e.g., TS over
        symbol vs CS over date) and splits the execution into sequential steps.

        Reference: Polars Issue #25691
        (https://github.com/pola-rs/polars/issues/25691) Polars treats nested
        `.over()` calls as hierarchical partitions (inner within outer). However,
        in factor calculations, TS (Time-Series) and CS (Cross-Sectional)
        operations are often orthogonal, not hierarchical. To correctly compute
        `CS(TS(...))` or `TS(CS(...))`, we must materialize the inner result
        first (e.g., via `with_columns()`).

        Returns a list of (var_name, expr_str) tuples.
        The last item's var_name is 'result'.
        """
        expr_str = expr_str.strip()
        try:
            tree = ast.parse(expr_str, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {expr_str}") from e

        steps = []
        counter = 0  # Use nonlocal or simple var

        def _get_category(func_name: str) -> str:
            return OP_CATEGORY.get(func_name, "EL")

        def _subtree_contains_category(node: ast.AST, category: str) -> bool:
            """Return whether a subtree contains the target operator category."""
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                if not isinstance(child.func, ast.Name):
                    continue
                if _get_category(child.func.id) == category:
                    return True
            return False

        # We need a custom transformer that populates 'steps'
        class PlanTransformer(ast.NodeTransformer):
            def visit_Call(self, node: ast.Call) -> ast.AST:
                nonlocal counter
                # First visit children to handle deepest nesting first
                self.generic_visit(node)

                if not isinstance(node.func, ast.Name):
                    return node

                func_name = node.func.id
                cat = _get_category(func_name)

                # Check arguments for mismatch
                new_args: List[ast.expr] = []
                for arg in node.args:
                    conflict = False
                    if cat == "CS":
                        conflict = _subtree_contains_category(arg, "TS")
                    elif cat == "TS":
                        conflict = _subtree_contains_category(arg, "CS")

                    if conflict:
                        try:
                            arg_str = ast.unparse(arg)
                        except AttributeError:
                            raise RuntimeError("Python 3.9+ required for ast.unparse")

                        step_var = f"_auto_var_{counter}"
                        counter += 1
                        steps.append((step_var, arg_str))
                        new_args.append(ast.Name(id=step_var, ctx=ast.Load()))
                    else:
                        new_args.append(arg)

                node.args = new_args  # type: ignore
                return node

        # Transform the tree
        transformer = PlanTransformer()
        new_tree = transformer.visit(tree.body)  # Visit body directly

        # Final expression
        try:
            final_expr = ast.unparse(new_tree)
        except AttributeError:
            raise RuntimeError("Python 3.9+ required for ast.unparse")

        steps.append(("result", final_expr))

        return steps

    @staticmethod
    def _as_expr(value: Any) -> pl.Expr:
        """Wrap Python constants so logical operators always act on expressions."""
        if isinstance(value, pl.Expr):
            return value
        return pl.lit(value)

    @staticmethod
    def _chained_comparison_message(node: ast.Compare) -> str:
        """Explain a chained comparison, which is usually a precedence slip."""
        operands = [node.left, *node.comparators]
        bitwise = any(
            isinstance(operand, ast.BinOp)
            and isinstance(operand.op, (ast.BitAnd, ast.BitOr))
            for operand in operands
        )
        if bitwise:
            return (
                "Chained comparisons not supported: & and | bind tighter than "
                "comparisons, so wrap each comparison in parentheses, "
                "e.g. (Close > Open) & (Volume > 0), or use and / or instead."
            )
        return (
            "Chained comparisons not supported: split into two comparisons "
            "joined by & (or and), e.g. (1 < Close) & (Close < 5)."
        )

    def _visit(self, node: Any) -> Any:
        # Handle Function Calls
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are supported")
            func_name = node.func.id

            if func_name not in OPS_MAP:
                raise ValueError(f"Unknown function: {func_name}")

            args = [self._visit(arg) for arg in node.args]
            return OPS_MAP[func_name](*args)

        # Handle Variables (Columns)
        elif isinstance(node, ast.Name):
            # Convention: Variable names are column names.
            # Convert "Close" -> "close" to match standard lowercase columns
            return pl.col(node.id.lower())

        # Handle Constants
        elif isinstance(node, ast.Constant):
            return node.value

        # Handle Binary Operators (+, -, *, /)
        elif isinstance(node, ast.BinOp):
            left = self._visit(node.left)
            right = self._visit(node.right)

            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                return left / right
            elif isinstance(node.op, ast.Pow):
                return left**right
            elif isinstance(node.op, ast.Mod):
                return left % right
            elif isinstance(node.op, ast.BitAnd):
                return left & right
            elif isinstance(node.op, ast.BitOr):
                return left | right
            else:
                raise ValueError(f"Unsupported binary operator: {type(node.op)}")

        # Handle Logical Operators (and, or)
        # Element-wise like & and |: no short-circuit, nulls follow
        # three-valued logic.
        elif isinstance(node, ast.BoolOp):
            values = [self._as_expr(self._visit(value)) for value in node.values]
            combined = values[0]
            for value in values[1:]:
                if isinstance(node.op, ast.And):
                    combined = combined & value
                else:
                    combined = combined | value
            return combined

        # Handle Unary Operators (-, not, ~)
        elif isinstance(node, ast.UnaryOp):
            operand = self._visit(node.operand)

            if isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, (ast.Not, ast.Invert)):
                return ~self._as_expr(operand)  # Logical NOT
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")

        # Handle Comparisons (<, >, ==)
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ValueError(self._chained_comparison_message(node))

            left = self._visit(node.left)
            right = self._visit(node.comparators[0])
            op = node.ops[0]

            if isinstance(op, ast.Lt):
                return left < right
            elif isinstance(op, ast.LtE):
                return left <= right
            elif isinstance(op, ast.Gt):
                return left > right
            elif isinstance(op, ast.GtE):
                return left >= right
            elif isinstance(op, ast.Eq):
                return left == right
            elif isinstance(op, ast.NotEq):
                return left != right
            else:
                raise ValueError(f"Unsupported comparison operator: {type(op)}")

        # Handle Conditional Expression (IfExp): x if cond else y
        # Mapped to pl.when(cond).then(x).otherwise(y)
        elif isinstance(node, ast.IfExp):
            test = self._visit(node.test)
            body = self._visit(node.body)
            orelse = self._visit(node.orelse)
            return pl.when(test).then(body).otherwise(orelse)

        raise ValueError(f"Unsupported expression node type: {type(node)}")
