"""
RunwayGuard AI — Application Entrypoint.

Strictly an entrypoint with zero business logic.
All application logic is decoupled into src/ and src/ui/.
"""
from src.ui.blocks import create_ui

demo = create_ui()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    )
