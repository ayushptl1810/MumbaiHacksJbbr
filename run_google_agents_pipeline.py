"""Run Google Agents SDK Pipeline - Launcher script for the orchestrator"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator_agent import run_google_agents_orchestrator

def main():
    """Main launcher for Google Agents SDK pipeline"""
    print("🔮 Google Agents SDK Pipeline Launcher")
    print("=" * 50)
    print("🤖 Advanced AI orchestration with Google Agents")
    print("📡 Trend scanning + AI summarization + fact-checking")
    print()
    
    try:
        # Run the orchestrator
        exit_code = asyncio.run(run_google_agents_orchestrator())
        
        if exit_code == 0:
            print("\n🎉 Google Agents pipeline completed successfully!")
        else:
            print("\n❌ Google Agents pipeline failed!")
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline stopped by user")
        return 130
    except Exception as e:
        print(f"\n💥 Pipeline crashed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())