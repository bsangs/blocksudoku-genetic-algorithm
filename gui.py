import tkinter as tk
from tkinter import messagebox
from game import Game
from ga import GeneticAlgorithm
import threading
import time

CELL_SIZE = 30  # Adjusted for visibility
PADDING = 10
GAMES_PER_ROW = 5  # To display up to 10 games

class GameFrame(tk.Frame):
    def __init__(self, master, game_id, game, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.game_id = game_id
        self.game = game
        self.selected_block = None
        self.shadow_cells = []
        self.removing_lines = False

        # Initial color setup
        self.game.change_all_block_colors()

        # Create canvas
        self.canvas_width = self.game.board_size * CELL_SIZE + 2 * PADDING
        self.canvas_height = self.game.board_size * CELL_SIZE + 100  # Extra space for current blocks
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='white', highlightthickness=1, highlightbackground="black")
        self.canvas.pack()

        # Label for Game ID and Score
        self.label = tk.Label(self, text=f"Game {self.game_id} | Score: {self.game.get_score()}", font=("Arial", 10, "bold"))
        self.label.pack()

        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.on_mouse_leave)

        # Initial draw
        self.draw()

    def draw(self):
        self.canvas.delete("all")
        self.draw_board()
        self.draw_placed_blocks()
        self.draw_current_blocks()
        self.draw_shadow()
        self.draw_score()
        if self.game.game_over:
            self.canvas.create_text(
                self.canvas_width / 2,
                self.canvas_height - 20,
                text="게임 오버",
                font=("Arial", 16, "bold"),
                fill="red"
            )

    def draw_board(self):
        # Draw grid lines
        for i in range(self.game.board_size +1):
            # Horizontal lines
            self.canvas.create_line(
                PADDING,
                PADDING + i * CELL_SIZE,
                PADDING + self.game.board_size * CELL_SIZE,
                PADDING + i * CELL_SIZE
            )
            # Vertical lines
            self.canvas.create_line(
                PADDING + i * CELL_SIZE,
                PADDING,
                PADDING + i * CELL_SIZE,
                PADDING + self.game.board_size * CELL_SIZE
            )

    def draw_placed_blocks(self):
        for x in range(self.game.board_size):
            for y in range(self.game.board_size):
                block = self.game.board[x][y]
                if block:
                    color = self.game.get_block_color(block)
                    self.canvas.create_rectangle(
                        PADDING + y * CELL_SIZE,
                        PADDING + x * CELL_SIZE,
                        PADDING + (y +1) * CELL_SIZE,
                        PADDING + (x +1) * CELL_SIZE,
                        fill=color,
                        outline='black'
                    )

    def draw_current_blocks(self):
        # Draw area for current blocks
        start_x = PADDING
        start_y = PADDING + self.game.board_size * CELL_SIZE + 10
        self.canvas.create_text(start_x, start_y -20, anchor='nw', text="블럭:", font=("Arial", 10, "bold"))
        for idx, block in enumerate(self.game.get_current_blocks()):
            block_x = start_x + idx * (CELL_SIZE + 10)
            block_y = start_y
            self.canvas.create_rectangle(
                block_x,
                block_y,
                block_x + CELL_SIZE,
                block_y + CELL_SIZE,
                fill='lightgrey',
                outline='black'
            )
            # Draw the shape scaled to fit
            for cell in block.shape:
                cell_size = CELL_SIZE // 3
                cx = block_x + 5 + cell[1] * cell_size
                cy = block_y + 5 + cell[0] * cell_size
                self.canvas.create_rectangle(
                    cx,
                    cy,
                    cx + cell_size -2,
                    cy + cell_size -2,
                    fill='black'
                )

    def draw_shadow(self):
        if self.selected_block and self.shadow_cells:
            for x, y in self.shadow_cells:
                self.canvas.create_rectangle(
                    PADDING + y * CELL_SIZE,
                    PADDING + x * CELL_SIZE,
                    PADDING + (y +1) * CELL_SIZE,
                    PADDING + (x +1) * CELL_SIZE,
                    fill='grey',
                    stipple='gray50'
                )

    def draw_score(self):
        self.label.config(text=f"Game {self.game_id} | Score: {self.game.get_score()}")

    def on_click(self, event):
        if self.game.game_over or self.removing_lines:
            return

        x, y = event.x, event.y
        # Check if a current block was clicked
        clicked_block = self.get_block_at_current_blocks(x, y)
        if clicked_block:
            # Select or deselect the block
            if self.selected_block == clicked_block:
                self.selected_block = None
                self.shadow_cells = []
            else:
                self.selected_block = clicked_block
            self.draw()
            return

        # Attempt to place the selected block
        if self.selected_block:
            board_x, board_y = self.get_board_position(x, y)
            if board_x is not None and board_y is not None:
                placed, removed_cells = self.game.place_block(self.selected_block, board_x, board_y)
                if placed:
                    if removed_cells > 0:
                        self.removing_lines = True
                        self.canvas.after(500, self.after_remove_lines)
                    else:
                        self.draw()
                    self.selected_block = None
                    self.shadow_cells = []
                    if self.game.game_over:
                        messagebox.showinfo("게임 오버", f"Game {self.game_id} 게임이 종료되었습니다.")
                else:
                    messagebox.showerror("오류", "블록을 놓을 수 없습니다.")

    def after_remove_lines(self):
        self.removing_lines = False
        self.draw()

    def on_mouse_move(self, event):
        if not self.selected_block or self.game.game_over:
            self.shadow_cells = []
            self.draw()
            return
        x, y = event.x, event.y
        board_x, board_y = self.get_board_position(x, y)
        if board_x is not None and board_y is not None:
            # Tentative placement position
            tentative_cells = self.selected_block.get_cells(board_x, board_y)
            # Validate
            valid = True
            for cell_x, cell_y in tentative_cells:
                if cell_x < 0 or cell_x >= self.game.board_size or cell_y < 0 or cell_y >= self.game.board_size:
                    valid = False
                    break
                if self.game.board[cell_x][cell_y] is not None:
                    valid = False
                    break
            if valid:
                self.shadow_cells = tentative_cells
            else:
                self.shadow_cells = []  # Invalid placement
        else:
            self.shadow_cells = []
        self.draw()

    def on_mouse_leave(self, event):
        self.shadow_cells = []
        self.draw()

    def get_block_at_current_blocks(self, x, y):
        start_x = PADDING
        start_y = PADDING + self.game.board_size * CELL_SIZE + 10
        for idx, block in enumerate(self.game.get_current_blocks()):
            block_x = start_x + idx * (CELL_SIZE + 10)
            block_y = start_y
            if block_x <= x <= block_x + CELL_SIZE and block_y <= y <= block_y + CELL_SIZE:
                return block
        return None

    def get_board_position(self, x, y):
        """
        Converts pixel coordinates to board grid coordinates.
        """
        if PADDING <= x < PADDING + self.game.board_size * CELL_SIZE and PADDING <= y < PADDING + self.game.board_size * CELL_SIZE:
            board_x = (y - PADDING) // CELL_SIZE
            board_y = (x - PADDING) // CELL_SIZE
            return int(board_x), int(board_y)
        return None, None

    def update_game(self):
        with self.game.lock:
            self.draw()


class ControlPanel(tk.Frame):
    def __init__(self, master, ga, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.ga = ga
        self.is_running = False

        small_font = ("Arial", 8)

        self.start_button = tk.Button(self, text="시작", command=self.start_ga, font=small_font)
        self.start_button.grid(row=0, column=0, padx=2, pady=2)

        self.stop_button = tk.Button(self, text="멈춤", command=self.stop_ga, state="disabled", font=small_font)
        self.stop_button.grid(row=0, column=1, padx=2, pady=2)

        self.save_button = tk.Button(self, text="저장", command=self.ga.save_population, font=small_font)
        self.save_button.grid(row=0, column=2, padx=2, pady=2)

        self.load_button = tk.Button(self, text="불러오기", command=self.ga.load_population, font=small_font)
        self.load_button.grid(row=0, column=3, padx=2, pady=2)

        self.generation_label = tk.Label(self, text="세대: 0", font=small_font)
        self.generation_label.grid(row=0, column=4, padx=5)

        self.best_score_label = tk.Label(self, text="최고 점수: 0", font=small_font)
        self.best_score_label.grid(row=0, column=5, padx=5)

        self.average_score_label = tk.Label(self, text="평균 점수: 0", font=small_font)
        self.average_score_label.grid(row=0, column=6, padx=5)

        self.min_score_label = tk.Label(self, text="최소 점수: 0", font=small_font)
        self.min_score_label.grid(row=0, column=7, padx=5)

    def start_ga(self):
        if not self.is_running:
            self.is_running = True
            self.ga.start()
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")

    def stop_ga(self):
        if self.is_running:
            self.ga.stop()
            self.is_running = False
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")

    def update_labels(self, generation, best_score, average_score, min_score):
        self.generation_label.config(text=f"세대: {generation}")
        self.best_score_label.config(text=f"최고 점수: {best_score}")
        self.average_score_label.config(text=f"평균 점수: {average_score:.2f}")
        self.min_score_label.config(text=f"최소 점수: {min_score}")

class GUI:
    def __init__(self, root, total_games=10):
        self.root = root
        self.root.title("블록 스도쿠 게임 - 유전 알고리즘용 10개 게임")
        self.total_games = total_games
        self.game_frames = []

        

        # Create Game instances for GA and GUI
        self.games = [Game(strategy=None) for _ in range(total_games)]

        # Create GA instance
        self.ga = GeneticAlgorithm(pop_size=total_games, generations=1000000, games=self.games, gui=self)

        # Control Panel을 상단에 배치
        self.control_panel = ControlPanel(root, self.ga)
        self.control_panel.pack(side="top", fill="x")

        # Create a scrollable frame to hold multiple games
        self.canvas = tk.Canvas(root, borderwidth=0, background="#ffffff")
        self.frame = tk.Frame(self.canvas, background="#ffffff")
        self.vsb = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((4,4), window=self.frame, anchor="nw", tags="self.frame")

        self.frame.bind("<Configure>", self.on_frame_configure)

        self.setup_games()

        # Start GUI update loop
        self.update_gui()

    def setup_games(self):
        for i in range(1, self.total_games +1):
            game_frame = GameFrame(self.frame, i, self.games[i-1], borderwidth=2, relief="groove")
            row = (i -1) // GAMES_PER_ROW
            col = (i -1) % GAMES_PER_ROW
            game_frame.grid(row=row, column=col, padx=5, pady=5)
            self.game_frames.append(game_frame)

    def on_frame_configure(self, event):
        # Reset the scroll region to encompass the inner frame
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_gui(self):
        # Update game frames
        for frame in self.game_frames:
            frame.update_game()
        # 잠금 내에서 필요한 데이터만 가져오기
        with self.ga.lock:
            generation = self.ga.current_generation
            best_score = self.ga.best_score
            average_score = self.ga.average_score
            min_score = self.ga.min_score
        # 잠금 해제 후 라벨 업데이트
        self.control_panel.update_labels(generation, best_score, average_score, min_score)
        # Schedule next update
        self.root.after(100, self.update_gui)  # Update every 100 milliseconds

    def run(self):
        self.root.mainloop()
