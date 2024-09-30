import tkinter as tk
from gui import GUI

def main():
    root = tk.Tk()
    gui = GUI(root, total_games=10)  # Display 10 games
    gui.run()

if __name__ == "__main__":
    main()
