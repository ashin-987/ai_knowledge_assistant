#!/usr/bin/env python3
"""
Migration Script: Basic → Improved RAG System
Automates the upgrade process with safety checks
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

class RAGSystemMigration:
    def __init__(self):
        self.backup_dir = f"./backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.issues = []
        self.warnings = []
        
    def print_banner(self):
        """Print welcome banner."""
        print("\n" + "="*70)
        print("  🚀 RAG SYSTEM MIGRATION TOOL")
        print("  Basic System → Improved System")
        print("="*70 + "\n")
    
    def check_prerequisites(self):
        """Check if system is ready for migration."""
        print("📋 Checking prerequisites...\n")
        
        # Check Python version
        import sys
        version = sys.version_info
        if version.major >= 3 and version.minor >= 9:
            print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        else:
            self.issues.append(f"Python 3.9+ required (found {version.major}.{version.minor})")
            print(f"❌ Python version: {version.major}.{version.minor} (need 3.9+)")
        
        # Check for required new files
        required_files = [
            'vector_store_improved.py',
            'document_processor_improved.py',
            'rag_engine_improved.py',
            'app_improved.py',
            'requirements_improved.txt'
        ]
        
        for file in required_files:
            if Path(file).exists():
                print(f"✅ Found: {file}")
            else:
                self.issues.append(f"Missing required file: {file}")
                print(f"❌ Missing: {file}")
        
        # Check for old files
        old_files = [
            'vector_store.py',
            'document_processor.py',
            'rag_engine.py',
            'app.py'
        ]
        
        print("\n📦 Old system files:")
        for file in old_files:
            if Path(file).exists():
                print(f"   Found: {file}")
        
        # Check for existing database
        if Path('./chroma_db').exists():
            print(f"\n📊 Found existing database: ./chroma_db")
            self.warnings.append("Existing database found - will be backed up")
        
        # Check for API key
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key and api_key.startswith("gsk_"):
            print(f"\n✅ API Key found: {api_key[:15]}...")
        else:
            self.warnings.append("GROQ_API_KEY not found in .env")
            print(f"\n⚠️  No valid API key in .env")
        
        return len(self.issues) == 0
    
    def create_backup(self):
        """Backup existing system."""
        print(f"\n💾 Creating backup in: {self.backup_dir}\n")
        
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Backup database
        if Path('./chroma_db').exists():
            shutil.copytree('./chroma_db', f'{self.backup_dir}/chroma_db')
            print(f"✅ Backed up: chroma_db/")
        
        # Backup documents
        if Path('./uploaded_documents').exists():
            shutil.copytree('./uploaded_documents', f'{self.backup_dir}/uploaded_documents')
            print(f"✅ Backed up: uploaded_documents/")
        
        # Backup old code files
        old_files = ['vector_store.py', 'document_processor.py', 'rag_engine.py', 'app.py', 'requirements.txt']
        for file in old_files:
            if Path(file).exists():
                shutil.copy(file, f'{self.backup_dir}/{file}')
                print(f"✅ Backed up: {file}")
        
        # Backup .env if exists
        if Path('.env').exists():
            shutil.copy('.env', f'{self.backup_dir}/.env')
            print(f"✅ Backed up: .env")
        
        print(f"\n💚 Backup complete!")
    
    def install_dependencies(self):
        """Install improved system dependencies."""
        print(f"\n📦 Installing dependencies...\n")
        
        import subprocess
        
        try:
            result = subprocess.run(
                ['pip', 'install', '-r', 'requirements_improved.txt'],
                capture_output=True,
                text=True,
                check=True
            )
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies:")
            print(e.stderr)
            self.issues.append("Dependency installation failed")
            return False
    
    def setup_configuration(self):
        """Set up configuration files."""
        print(f"\n⚙️  Setting up configuration...\n")
        
        # Create .env if it doesn't exist
        if not Path('.env').exists():
            print("Creating .env template...")
            with open('.env', 'w') as f:
                f.write("# Groq API Configuration\n")
                f.write("GROQ_API_KEY=your_api_key_here\n\n")
                f.write("# Optional: Database path\n")
                f.write("# CHROMA_DB_PATH=./chroma_db\n")
            print("✅ Created .env template - PLEASE ADD YOUR API KEY!")
            self.warnings.append("Update .env with your GROQ_API_KEY")
        else:
            print("✅ .env already exists")
        
        # Create .streamlit/config.toml
        streamlit_dir = Path('.streamlit')
        streamlit_dir.mkdir(exist_ok=True)
        
        config_file = streamlit_dir / 'config.toml'
        if not config_file.exists():
            print("Creating Streamlit config...")
            with open(config_file, 'w') as f:
                f.write("""[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
maxMessageSize = 200
enableXsrfProtection = true

[browser]
gatherUsageStats = false
""")
            print("✅ Created .streamlit/config.toml")
        else:
            print("✅ Streamlit config already exists")
        
        # Create .gitignore if it doesn't exist
        if not Path('.gitignore').exists():
            print("Creating .gitignore...")
            with open('.gitignore', 'w') as f:
                f.write("""# Python
__pycache__/
*.py[cod]
*.so
.Python
venv/
.venv/

# Environment
.env
.env.local

# Database
chroma_db/
test_chroma_db/
*.db

# Streamlit
.streamlit/secrets.toml

# Test files
test_documents/
uploaded_documents/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
""")
            print("✅ Created .gitignore")
        else:
            print("✅ .gitignore already exists")
    
    def test_system(self):
        """Run basic system tests."""
        print(f"\n🧪 Testing improved system...\n")
        
        try:
            # Test imports
            print("Testing imports...")
            from vector_store import ImprovedVectorStore
            from document_processor import ImprovedDocumentProcessor
            from rag_engine import ImprovedRAGEngine
            print("✅ All imports successful")
            
            # Test document processor
            print("\nTesting document processor...")
            processor = ImprovedDocumentProcessor()
            test_text = "This is a test. This should be chunked properly."
            chunks = processor.chunk_text_semantic(test_text)
            if chunks:
                print(f"✅ Document processor works ({len(chunks)} chunks)")
            else:
                self.warnings.append("Document processor returned no chunks")
            
            # Test vector store initialization
            print("\nTesting vector store...")
            store = ImprovedVectorStore(persist_directory="./test_migration_db")
            stats = store.get_stats()
            print(f"✅ Vector store initialized ({stats['total_chunks']} existing chunks)")
            
            # Cleanup test
            if Path('./test_migration_db').exists():
                shutil.rmtree('./test_migration_db')
            
            print("\n✅ All tests passed!")
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            self.issues.append(f"System test failed: {str(e)}")
            return False
    
    def create_migration_report(self):
        """Create migration report."""
        report_file = f"{self.backup_dir}/migration_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RAG SYSTEM MIGRATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Backup Location: {self.backup_dir}\n\n")
            
            if self.issues:
                f.write("ISSUES:\n")
                for issue in self.issues:
                    f.write(f"  ❌ {issue}\n")
                f.write("\n")
            else:
                f.write("✅ No issues detected\n\n")
            
            if self.warnings:
                f.write("WARNINGS:\n")
                for warning in self.warnings:
                    f.write(f"  ⚠️  {warning}\n")
                f.write("\n")
            else:
                f.write("✅ No warnings\n\n")
            
            f.write("NEXT STEPS:\n")
            f.write("  1. Review this report\n")
            f.write("  2. Update .env with GROQ_API_KEY if needed\n")
            f.write("  3. Run: python test_improved.py\n")
            f.write("  4. Run: streamlit run app_improved.py\n")
            f.write("  5. Re-upload and process your documents\n\n")
            
            f.write("ROLLBACK (if needed):\n")
            f.write(f"  - Restore from: {self.backup_dir}/\n")
            f.write("  - Copy back old files and database\n\n")
        
        print(f"\n📄 Migration report saved: {report_file}")
    
    def run(self):
        """Run complete migration."""
        self.print_banner()
        
        # Step 1: Prerequisites
        if not self.check_prerequisites():
            print("\n❌ Prerequisites check failed!")
            print("\nIssues found:")
            for issue in self.issues:
                print(f"  • {issue}")
            print("\nPlease fix these issues and try again.")
            return False
        
        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")
            
            response = input("\nContinue with migration? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Migration cancelled.")
                return False
        
        # Step 2: Backup
        self.create_backup()
        
        # Step 3: Install dependencies
        print("\nReady to install dependencies?")
        response = input("Install now? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            if not self.install_dependencies():
                print("\n❌ Dependency installation failed!")
                print("You can install manually: pip install -r requirements_improved.txt")
        else:
            print("⏭️  Skipping dependency installation")
            print("Remember to run: pip install -r requirements_improved.txt")
        
        # Step 4: Configuration
        self.setup_configuration()
        
        # Step 5: Test
        self.test_system()
        
        # Step 6: Report
        self.create_migration_report()
        
        # Summary
        print("\n" + "="*70)
        print("  ✅ MIGRATION COMPLETE!")
        print("="*70 + "\n")
        
        print("📊 Summary:")
        print(f"  • Backup created: {self.backup_dir}/")
        print(f"  • Configuration files: Created")
        print(f"  • System tests: {'✅ Passed' if len(self.issues) == 0 else '❌ Failed'}")
        
        if self.warnings:
            print(f"\n⚠️  Action Required:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        print("\n🚀 Next Steps:")
        print("  1. Review migration report")
        print("  2. Run test suite: python test_improved.py")
        print("  3. Start app: streamlit run app_improved.py")
        print("  4. Upload and process your documents")
        
        print("\n💡 Tips:")
        print("  • The improved system uses better chunking")
        print("  • Consider re-processing documents for best results")
        print("  • Hybrid search provides better retrieval quality")
        print("  • Enable reranking for maximum precision")
        
        print("\n📚 Documentation:")
        print("  • README: README_IMPROVED.md")
        print("  • Comparison: COMPARISON.md")
        print("  • Deployment: DEPLOYMENT.md")
        
        return True

def main():
    """Main entry point."""
    migration = RAGSystemMigration()
    
    try:
        success = migration.run()
        
        if success:
            print("\n🎉 Migration successful! Enjoy your improved RAG system!")
        else:
            print("\n⚠️  Migration incomplete. Please review the issues above.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Migration interrupted by user")
        print(f"Backup available at: {migration.backup_dir}/")
    except Exception as e:
        print(f"\n❌ Migration failed with error: {e}")
        print(f"Backup available at: {migration.backup_dir}/")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
