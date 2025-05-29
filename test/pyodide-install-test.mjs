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
  `);
};

start();
