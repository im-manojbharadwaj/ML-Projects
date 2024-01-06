Step 1:
Install VS Code and Python Extension (https://marketplace.visualstudio.com/items?itemName=ms-python.python)

Step 2:
Install a version of Python 3
Options include:
    (All operating systems) A download from python.org; typically use the Download button that appears first on the page.
    (Linux) The built-in Python 3 installation works well, but to install other Python packages you must run sudo apt install python3-pip in the terminal.
    (macOS) An installation through Homebrew on macOS using brew install python3.
    (All operating systems) A download from Anaconda (for data science purposes).

Step 3:
On Windows, make sure the location of your Python interpreter is included in your PATH environment variable. You can check the location by running path at the command prompt. If the Python interpreter's folder isn't included, open Windows Settings, search for "environment", select Edit environment variables for your account, then edit the Path variable to include that folder.

Create a project environment for the Flask tutorial

Step 1:
On your file system, create a folder for this tutorial, such as pan-tamper

Step 2:
Open this folder in VS Code by navigating to the folder in a terminal and running code ., or by running VS Code and using the File > Open Folder command.

Step 3:
In VS Code, open the Command Palette (View > Command Palette or (⇧⌘P)). Then select the Python: Create Environment command to create a virtual environment in your workspace. Select venv and then the Python environment you want to use to create it.

Step 4:
After your virtual environment creation has been completed, run Terminal: Create New Terminal (⌃⇧`) from the Command Palette, which creates a terminal and automatically activates the virtual environment by running its activation script.
    Note: 
        On Windows, if your default terminal type is PowerShell, you may see an error that it cannot run activate.ps1 because running scripts is disabled on the system. The error provides a link for information on how to allow scripts. Otherwise, use Terminal: Select Default Profile to set "Command Prompt" or "Git Bash" as your default instead.

Step 5:
Install Flask in the virtual environment by running the following command in the VS Code Terminal: python -m pip install flask

Step 6:
You now have a self-contained environment ready for writing Flask code. VS Code activates the environment automatically when you use Terminal: Create New Terminal. If you open a separate command prompt or terminal, activate the environment by running source .venv/bin/activate (Linux/macOS) or .venv\Scripts\Activate.ps1 (Windows). You know the environment is activated when the command prompt shows (.venv) at the beginning.


Now use the files available in my repository to run your application in VS Code
