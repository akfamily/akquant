use akquant::stub_info;

fn main() -> pyo3_stub_gen::Result<()> {
    let stub = stub_info()?;
    println!("Generating stubs...");
    stub.generate()?;
    println!("Done.");
    Ok(())
}
