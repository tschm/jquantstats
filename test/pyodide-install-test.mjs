// test/pyodide-install-test.mjs

import { loadPyodide } from "pyodide";

const start = async () => {
  const pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/" });

  console.log("✅ Pyodide loaded");

  const wheel_url = "http://localhost:8000/" + require('fs').readdirSync('./dist').find(f => f.endsWith(".whl"));

  console.log(`Installing: ${wheel_url}`);
  await pyodide.loadPackage("micropip");
  await pyodide.runPythonAsync(`
    import micropip
    await micropip.install("${wheel_url}")
    import jquantstats
    print("✅ jquantstats successfully imported")

    # Test basic functionality
    import polars as pl
    from jquantstats.api import build_data

    # Create sample returns data
    returns = pl.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Asset1": [0.01, -0.02, 0.03]
    }).with_columns(pl.col("Date").str.to_date())

    # Basic usage
    data = build_data(returns=returns)

    # Calculate a statistic
    volatility = data.stats.volatility()

    print("✅ jquantstats functionality tested successfully")
  `);
};

start();
