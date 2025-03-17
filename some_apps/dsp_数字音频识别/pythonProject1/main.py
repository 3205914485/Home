import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox
import dtw_slot

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Audio Matching')
        self.setGeometry(1000, 500, 640, 480)
        self.mode = 'lib'
        # Create buttons
        self.record_template_button = QPushButton('Record Templates', self)
        self.record_test_button = QPushButton('Record Test Audio', self)
        self.exit_button = QPushButton('Exit', self)

        # Set button actions
        self.record_template_button.clicked.connect(self.record_templates)
        self.record_test_button.clicked.connect(self.record_test_audio)
        self.exit_button.clicked.connect(self.close_application)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.record_template_button)
        layout.addWidget(self.record_test_button)
        layout.addWidget(self.exit_button)
        self.setLayout(layout)

    def record_templates(self):
        num_records = 10  # Number of recordings
        template_dir = 'templates/'

        for i in range(num_records):
            self.show_message(f"Recording template {i + 1}/{num_records}...")
            recorded_audio = dtw_slot.record_audio()
            dtw_slot.save_audio(recorded_audio, i, template_dir)

        self.show_message("Template recording complete! 10 files saved.")
        self.mode = 'self'

    def record_test_audio(self):
        sample_dir = 'sample/'

        recorded_audio = dtw_slot.record_audio()
        fil_nam = dtw_slot.save_audio(recorded_audio, 0, sample_dir)

        self.show_message("Test audio recorded. Now matching with templates...")

        # Match the newly recorded test audio with templates
        match_result = dtw_slot.match_audio(fil_nam, self.mode)
        
        # Display the match result
        self.show_message(f"Matching result: {match_result}")

    def close_application(self):
        QApplication.quit()

    def show_message(self, message):
        """Utility function to display messages to the user."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Information")
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
