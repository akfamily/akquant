(function () {
  "use strict";

  var ENGLISH_PREFIX = /^\/en(?:\/|$)/;
  var RESULT_LIST_SELECTOR = ".md-search-result__list";
  var RESULT_ITEM_SELECTOR = ".md-search-result__item";
  var RESULT_LINK_SELECTOR = ".md-search-result__link";
  var SEARCH_QUERY_SELECTOR = 'input[data-md-component="search-query"]';
  var EMPTY_STATE_ATTR = "data-akquant-locale-empty";
  var rafId = null;

  function getCurrentLocale() {
    return ENGLISH_PREFIX.test(window.location.pathname) ? "en" : "zh";
  }

  function getResultPath(href) {
    try {
      return new URL(href, window.location.origin).pathname;
    } catch (error) {
      return "";
    }
  }

  function matchesCurrentLocale(pathname, locale) {
    if (!pathname) {
      return true;
    }
    return locale === "en"
      ? ENGLISH_PREFIX.test(pathname)
      : !ENGLISH_PREFIX.test(pathname);
  }

  function getQueryText() {
    var input = document.querySelector(SEARCH_QUERY_SELECTOR);
    return input ? input.value.trim() : "";
  }

  function getEmptyMessage(locale) {
    return locale === "en"
      ? "No results in the current language. Switch language to search the translated page."
      : "当前语言暂无匹配结果，可切换语言后重试。";
  }

  function ensureEmptyState(list, locale, shouldShow) {
    var existing = list.querySelector("[" + EMPTY_STATE_ATTR + "]");

    if (!shouldShow) {
      if (existing) {
        existing.remove();
      }
      return;
    }

    if (existing) {
      existing.querySelector("p").textContent = getEmptyMessage(locale);
      return;
    }

    var item = document.createElement("li");
    item.className = "md-search-result__item";
    item.setAttribute(EMPTY_STATE_ATTR, "true");

    var article = document.createElement("article");
    article.className = "md-search-result__article md-typeset";

    var paragraph = document.createElement("p");
    paragraph.textContent = getEmptyMessage(locale);

    article.appendChild(paragraph);
    item.appendChild(article);
    list.appendChild(item);
  }

  function filterList(list) {
    var locale = getCurrentLocale();
    var items = Array.prototype.filter.call(
      list.querySelectorAll(RESULT_ITEM_SELECTOR),
      function (item) {
        return !item.hasAttribute(EMPTY_STATE_ATTR);
      }
    );

    if (!items.length) {
      ensureEmptyState(list, locale, false);
      return;
    }

    var visibleCount = 0;

    items.forEach(function (item) {
      var links = item.querySelectorAll(RESULT_LINK_SELECTOR);
      var keep = Array.prototype.some.call(links, function (link) {
        var href = link.getAttribute("href") || "";
        return matchesCurrentLocale(getResultPath(href), locale);
      });

      item.hidden = !keep;
      if (keep) {
        visibleCount += 1;
      }
    });

    ensureEmptyState(
      list,
      locale,
      visibleCount === 0 && items.length > 0 && getQueryText().length > 0
    );
  }

  function filterAllSearchResults() {
    document.querySelectorAll(RESULT_LIST_SELECTOR).forEach(filterList);
  }

  function scheduleFilter() {
    if (rafId !== null) {
      window.cancelAnimationFrame(rafId);
    }
    rafId = window.requestAnimationFrame(function () {
      rafId = null;
      filterAllSearchResults();
    });
  }

  document.addEventListener("input", function (event) {
    if (event.target && event.target.matches(SEARCH_QUERY_SELECTOR)) {
      scheduleFilter();
    }
  });

  window.addEventListener("hashchange", scheduleFilter);
  window.addEventListener("pageshow", scheduleFilter);
  document.addEventListener("DOMContentLoaded", scheduleFilter);

  var observer = new MutationObserver(function () {
    scheduleFilter();
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });
})();
