from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
import VGA_wrapped
import threading
import sys
import time

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VGA Agent")
        self.setFixedSize(500, 300)
        self.setStyleSheet("background-color: black; color: lime;")
        layout = QVBoxLayout()
        self.label = QLabel("Initializing VGA Agent...")
        self.label.setStyleSheet("font-size: 20px;")
        layout.addWidget(self.label)
        self.setLayout(layout)

    def update_status(self, text):
        self.label.setText(text)
        QApplication.processEvents()

def run_agent_and_close_ui(ui):
    ui.update_status("Loading Modules...")
    time.sleep(2)
    ui.update_status("Spawning Agent Brain...")
    time.sleep(2)
    ui.update_status("Starting Game Interface...")
    time.sleep(2)
    VGA_wrapped.run()
    app.quit()

app = QApplication(sys.argv)
window = SplashScreen()
window.show()

threading.Thread(target=run_agent_and_close_ui, args=(window,)).start()
sys.exit(app.exec_())
