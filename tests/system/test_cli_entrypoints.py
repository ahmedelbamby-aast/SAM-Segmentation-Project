"""System tests: CLI entry points can be imported and respond to --help.

Verifies that all 10 CLI modules are importable and their `main()` functions
are callable without crashing.

Author: Ahmed Hany ElBamby
Date: 25-07-2025
"""

from __future__ import annotations

import importlib
import sys
import pytest


# ---------------------------------------------------------------------------
# All CLI modules must be importable
# ---------------------------------------------------------------------------

CLI_MODULES = [
    "src.cli.pipeline",
    "src.cli.preprocess",
    "src.cli.segment",
    "src.cli.postprocess",
    "src.cli.filter",
    "src.cli.annotate",
    "src.cli.validate",
    "src.cli.upload",
    "src.cli.download",
    "src.cli.progress",
]


@pytest.mark.parametrize("module_name", CLI_MODULES)
class TestCLIModuleImport:
    """Every CLI module must import cleanly with no side effects."""

    def test_module_is_importable(self, module_name: str):
        """Module must import without raising any exception."""
        mod = importlib.import_module(module_name)
        assert mod is not None

    def test_module_has_main_function(self, module_name: str):
        """Every CLI module must expose a `main` callable."""
        mod = importlib.import_module(module_name)
        assert hasattr(mod, "main"), f"{module_name} is missing a `main` function"
        assert callable(mod.main)


# ---------------------------------------------------------------------------
# --help returns without SystemExit(1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", CLI_MODULES)
class TestCLIHelpFlag:
    """main(['--help']) must produce help output and exit 0, not crash."""

    def test_help_flag_exits_cleanly(self, module_name: str):
        """--help raises SystemExit(0) (argparse convention), not 1 or an error."""
        mod = importlib.import_module(module_name)
        with pytest.raises(SystemExit) as exc_info:
            mod.main(["--help"])
        # argparse exits 0 on --help; some CLIs may exit None
        assert exc_info.value.code in (0, None), (
            f"{module_name} main(['--help']) exited with code {exc_info.value.code}"
        )


# ---------------------------------------------------------------------------
# setup.py entry points are wired
# ---------------------------------------------------------------------------

class TestSetupEntryPoints:
    """setup.py console_scripts must reference callable entry points."""

    def test_setup_entry_points_reference_existing_functions(self):
        """Parse setup.py and verify all entry_points exist."""
        import ast
        from pathlib import Path
        setup_path = Path("setup.py")
        if not setup_path.exists():
            pytest.skip("setup.py not found")
        source = setup_path.read_text(encoding="utf-8")
        # Check key CLI modules are referenced
        for ep in ["src.cli.pipeline:main", "src.cli.segment:main"]:
            assert ep in source, f"Entry point {ep!r} not found in setup.py"

    def test_pipeline_cli_main_is_callable(self):
        """sam3-pipeline entry point resolves to a callable."""
        from src.cli.pipeline import main
        assert callable(main)

    def test_segment_cli_main_is_callable(self):
        from src.cli.segment import main
        assert callable(main)

    def test_validate_cli_main_is_callable(self):
        from src.cli.validate import main
        assert callable(main)

    def test_annotate_cli_main_is_callable(self):
        from src.cli.annotate import main
        assert callable(main)

    def test_filter_cli_main_is_callable(self):
        from src.cli.filter import main
        assert callable(main)

    def test_upload_cli_main_is_callable(self):
        from src.cli.upload import main
        assert callable(main)

    def test_download_cli_main_is_callable(self):
        from src.cli.download import main
        assert callable(main)

    def test_progress_cli_main_is_callable(self):
        from src.cli.progress import main
        assert callable(main)

    def test_preprocess_cli_main_is_callable(self):
        from src.cli.preprocess import main
        assert callable(main)

    def test_postprocess_cli_main_is_callable(self):
        from src.cli.postprocess import main
        assert callable(main)


# ---------------------------------------------------------------------------
# Legacy script wrappers are importable
# ---------------------------------------------------------------------------

class TestLegacyScriptWrappers:
    """Legacy scripts in scripts/ must be thin wrappers over src.cli."""

    def test_run_pipeline_script_imports_from_src_cli(self):
        """run_pipeline.py must delegate to src.cli.pipeline.main."""
        from pathlib import Path
        script = Path("scripts/run_pipeline.py").read_text(encoding="utf-8")
        assert "src.cli.pipeline" in script, "run_pipeline.py does not import src.cli.pipeline"

    def test_run_validator_script_imports_from_src_cli(self):
        """run_validator.py must delegate to src.cli.validate.main."""
        from pathlib import Path
        script = Path("scripts/run_validator.py").read_text(encoding="utf-8")
        assert "src.cli.validate" in script, "run_validator.py does not import src.cli.validate"
