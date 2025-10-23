"""
Setup Script for RAG Project
============================

This script helps beginners set up the RAG project step by step.

Run this script first before using the RAG system.
"""

import os
import sys
import subprocess

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_step(step_num, text):
    """Print a formatted step"""
    print(f"\n🔹 Step {step_num}: {text}")

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def install_packages():
    """Install required packages"""
    print("📦 Installing required packages...")
    print("This may take a few minutes...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error installing packages. Please check your internet connection.")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt file not found!")
        return False

def check_env_file():
    """Check if .env file exists and guide user"""
    print("🔑 Checking environment configuration...")
    
    if os.path.exists(".env"):
        print("✅ .env file found!")
        
        # Check if it has the API token
        with open(".env", "r") as f:
            content = f.read()
            if "your_token_here" in content:
                print("⚠️  Warning: You still need to add your actual HuggingFace API token!")
                print("📝 Edit the .env file and replace 'your_token_here' with your real token.")
                return False
            elif "HUGGINGFACE_API_TOKEN" in content:
                print("✅ HuggingFace API token configuration found!")
                return True
    else:
        print("❌ .env file not found!")
        print("📝 Creating .env file template...")
        
        # Copy template to .env
        if os.path.exists(".env.template"):
            with open(".env.template", "r") as template:
                with open(".env", "w") as env_file:
                    env_file.write(template.read())
            print("✅ .env file created from template!")
        else:
            # Create basic .env file
            with open(".env", "w") as f:
                f.write("# HuggingFace API Token\n")
                f.write("# Get your token from: https://huggingface.co/settings/tokens\n")
                f.write("HUGGINGFACE_API_TOKEN=your_token_here\n")
            print("✅ Basic .env file created!")
        
        print("\n🔗 To get your HuggingFace API token:")
        print("  1. Go to: https://huggingface.co/settings/tokens")
        print("  2. Create a new token (select 'Read' permission)")
        print("  3. Copy the token")
        print("  4. Open the .env file and replace 'your_token_here' with your token")
        
        return False

def check_pdf_files():
    """Check if PDF files exist"""
    print("📄 Checking for PDF files...")
    
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    
    if pdf_files:
        print(f"✅ Found {len(pdf_files)} PDF file(s):")
        for pdf in pdf_files:
            print(f"  - {pdf}")
        return True
    else:
        print("❌ No PDF files found!")
        print("📁 Please add a PDF file to this folder.")
        print("💡 The system is configured to use 'Chapter 01.pdf' by default.")
        return False

def run_test():
    """Run a quick test of the system"""
    print("🧪 Running quick test...")
    
    try:
        # Import test
        print("  Testing imports...")
        import dotenv
        print("  ✅ dotenv imported")
        
        # More imports would go here when packages are installed
        print("  ✅ Basic imports working!")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print_header("🚀 RAG Project Setup")
    
    print("Welcome to the RAG project setup!")
    print("This script will help you get everything ready.")
    
    success_count = 0
    total_steps = 5
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    if check_python_version():
        success_count += 1
    
    # Step 2: Install packages
    print_step(2, "Installing required packages")
    if install_packages():
        success_count += 1
    
    # Step 3: Check environment file
    print_step(3, "Setting up environment configuration")
    if check_env_file():
        success_count += 1
    
    # Step 4: Check PDF files
    print_step(4, "Checking for PDF documents")
    if check_pdf_files():
        success_count += 1
    
    # Step 5: Run test
    print_step(5, "Running system test")
    if run_test():
        success_count += 1
    
    # Summary
    print_header("📋 Setup Summary")
    print(f"✅ Completed: {success_count}/{total_steps} steps")
    
    if success_count == total_steps:
        print("\n🎉 Setup complete! You're ready to use the RAG system!")
        print("\n🚀 Next steps:")
        print("  1. Run: python rag_simple.py        (for a simple demo)")
        print("  2. Run: python rag_interactive.py   (for interactive chat)")
        print("  3. Run: python rag_advanced.py      (for advanced features)")
    else:
        print(f"\n⚠️  Setup incomplete. Please fix the issues above and run this script again.")
        print("\n🔧 Common fixes:")
        if success_count < 2:
            print("  - Make sure you have internet connection for package installation")
        if success_count < 3:
            print("  - Get your HuggingFace API token and add it to .env file")
        if success_count < 4:
            print("  - Add a PDF file to this folder")

if __name__ == "__main__":
    main()