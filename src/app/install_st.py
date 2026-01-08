import sys
import os
from streamlit.web import cli as stcli


def start():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    app_path = os.path.join(current_dir, 'app', 'web_app.py')

    print(f" Incerc sa pornesc aplicatia din: {app_path}")

    if not os.path.exists(app_path):
        print(f" EROARE: Nu gasesc fisierul web_app.py!")
        print(f"Am cautat aici: {app_path}")
        print("Te rog verifica daca ai creat fisierul 'web_app.py' in folderul 'src/app'.")
        return

    sys.argv = ["streamlit", "run", app_path]

    sys.exit(stcli.main())

if __name__ == "__main__":
    start()
