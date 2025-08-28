#!/usr/bin/env python3
"""
Simple installation test script for latent-design package.
Run this to test if your package can be installed and imported correctly.
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} successful!")
            return True
        else:
            print(f"❌ {description} failed!")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False


def test_imports():
    """Test if the package can be imported correctly."""
    print("\n🧪 Testing package imports...")
    
    try:
        import latent_design
        print(f"✅ Main package imported successfully! Version: {latent_design.__version__}")
        
        # Test key modules
        from latent_design.bases import Basis
        from latent_design.criteria import DOptimality
        from latent_design.models import FOF
        from latent_design.cli import main
        
        print("✅ All key modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False


def main():
    """Main installation test function."""
    print("🚀 Latent Design Labs - Installation Test")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ Error: pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Test installation
    success = True
    
    # Try to install in development mode
    if not run_command("uv pip install -e .", "Installing package in development mode"):
        success = False
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test CLI
    if not run_command("latent-design version", "Testing CLI version command"):
        success = False
    
    if not run_command("latent-design info", "Testing CLI info command"):
        success = False
    
    # Test running tests
    if not run_command("pytest tests/ -v", "Running tests"):
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Your package is working correctly.")
        print("\nYou can now:")
        print("  • Import the package: import latent_design")
        print("  • Use the CLI: latent-design --help")
        print("  • Install on other machines: pip install .")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
