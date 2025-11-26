from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QProgressBar
from PyQt5.QtCore import Qt
import VGA_wrapped
import threading
import sys

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VGA Agent Booting...")
        self.setFixedSize(500, 250)
        self.setStyleSheet("background-color: black; color: lime;")
        self.layout = QVBoxLayout()

        self.label = QLabel("Initializing VGA Agent...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 18px;")
        self.layout.addWidget(self.label)

        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #00ff00;
                border-radius: 5px;
                text-align: center;
                background-color: #1a1a1a;
            }
            QProgressBar::chunk {
                background-color: #00ff00;
                width: 10px;
            }
        """)
        self.layout.addWidget(self.progress)

        self.setLayout(self.layout)
        self.progress_step = 0

    def update_status(self, text):
        self.label.setText(text)
        self.progress_step = min(100, self.progress_step + 20)
        self.progress.setValue(self.progress_step)
        QApplication.processEvents()

def launch_agent():
    window.update_status("Loading OCR modules...")
    VGA_wrapped.run(status_callback=window.update_status)
    app.quit()

app = QApplication(sys.argv)
window = SplashScreen()
window.show()

thread = threading.Thread(target=launch_agent)
thread.start()

sys.exit(app.exec_())
