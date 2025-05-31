#from importlib.metadata import requires

#from packaging.requirements import Requirement

import jquantstats

# def get_runtime_requirements(package):
#     for req in requires(package) or []:
#         if req and 'extra == ' not in req:  # Skip any with extras (typically dev deps)
#             try:
#                 yield Requirement(req)
#             except:
#                 continue  # Skip malformed requirements

def test_version():
    assert jquantstats.__version__ is not None

# @pytest.mark.asyncio
# async def test_micropip_install(httpserver, tmp_path, resource_dir):
#     print(resource_dir.parent.parent.parent)
#     builder = build.ProjectBuilder(resource_dir.parent.parent.parent)
#
#     wheel_file = pathlib.Path(builder.build("wheel", tmp_path))
#
#     # Step 2: Serve the wheel over HTTP using pytest-httpserver
#     wheel_bytes = wheel_file.read_bytes()
#     httpserver.expect_request(f"/{wheel_file.name}").respond_with_data(
#     wheel_bytes, content_type="application/octet-stream")
#     wheel_url = httpserver.url_for(f"/{wheel_file.name}")
#
#     # Step 3: Install via micropip with dependency checking
#     import micropip
#     from importlib.metadata import version
#     import importlib
#
#     requirements = list(get_runtime_requirements())
#     assert requirements, "No runtime requirements found for jquantstats"
#     print(requirements)
#
#     # Install all requirements first (with exact versions)
#     for req in requirements:
#         print(str(req))
#         await micropip.install(str(req))
#         print(req.name)
#         importlib.import_module(req.name)
#         print(version(req.name))
#
#     package = micropip.list()
#     print(package)
#
#     # Step 4: Install via micropip
#     import micropip
#     await micropip.install(wheel_url)
#
#     # Step 4: Assert that it installed correctly
#     import jquantstats
#     assert hasattr(jquantstats, "__version__")
#
# async def test_dependencies():
#     import micropip
#
#     requirements = list(get_runtime_requirements(package="jquantstats"))
#     assert requirements, "No runtime requirements found for jquantstats"
#     print(requirements)
#
#     for req in requirements:
#         print(req)
#         await micropip.install(str(req))
#         # check the package has __version__
#         try:
#             module = importlib.import_module(req.name)
#
#             if not hasattr(module, "__version__"):
#                 print(f"Failed {req.name}: No version info")
#                 continue
#
#         except ImportError as e:
#             print(f"Failed Import for {e}")
#             continue
#
#         assert hasattr(module, "__version__")
#         print(module.__version__)
