import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.getcwd())

files_to_check = [
    "agents.graph",
    "agents.nodes",
    "agents.enhanced_nodes",
    "agents.query_understanding",
    "agents.semantic_query_interpreter",
    "agents.claim_level_grounding",
    "agents.evidence_gating",
    "agents.conversation_memory_retriever",
    "agents.conversation_state_reconstructor",
    "agents.scraper_node",
    "agents.prompts",
    "agents.state",
]

print('='*70)
print('CHECKING PYTHON FILES FOR IMPORT ISSUES')
print('='*70)

for module_name in files_to_check:
    print('\nChecking: {}'.format(module_name))
    print('-' * 70)
    try:
        mod = __import__(module_name, fromlist=[module_name.split('.')[-1]])
        print('OK: {} imported without errors'.format(module_name))
    except SyntaxError as e:
        print('FAIL: SYNTAX ERROR in {}:'.format(module_name))
        print('  Line {}: {}'.format(e.lineno, e.msg))
        if e.text:
            print('  {}'.format(e.text))
    except ImportError as e:
        print('FAIL: IMPORT ERROR in {}:'.format(module_name))
        print('  {}'.format(str(e)))
    except Exception as e:
        print('FAIL: ERROR in {}:'.format(module_name))
        print('  {}: {}'.format(type(e).__name__, str(e)))
        traceback.print_exc()

print('\n' + '='*70)
print('CHECKING COMPLETE')
print('='*70)
