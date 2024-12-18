import sys
from PyQt5 import QtWidgets, uic, QtGui, QtCore
import resources_rc
from stylesheet_helper import get_button_stylesheet, get_active_button_stylesheet
import pandas as pd
class MainScreen(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainScreen, self).__init__()
        uic.loadUi('main.ui', self)  # Load the UI file
        self.stackedWidget.setCurrentIndex(0)

        # Connect browse button to the function
        self.btnBrowse = self.findChild(QtWidgets.QPushButton, 'btnBrowse')  # Find the browse button by its name

        # Connect menu buttons to their functions
        self.btnHome.clicked.connect(lambda: self.on_menu_button_click('home'))
        self.btnData.clicked.connect(lambda: self.on_menu_button_click('data'))
        self.btnModel.clicked.connect(lambda: self.on_menu_button_click('model'))
        self.btnHelp.clicked.connect(lambda: self.on_menu_button_click('help'))
        self.btnBrowse.clicked.connect(self.open_file_dialog)
        self.btnNext.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))

        self.btnHome.setStyleSheet(get_active_button_stylesheet())
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(':/images/images/icons/icons8-home-30.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnHome.setIcon(icon)
        self.btnHome.setIconSize(QtCore.QSize(30, 30))

        self.dataTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)  # Adjust column widths

    def open_file_dialog(self):
        # Open file dialog to select a file
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        file_dialog.setNameFilter("CSV Files (*.csv);;Excel Files (*.xlsx)")
        
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.load_file_data(file_path)

    def load_file_data(self, file_path):
        try:
            # Determine file format
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")

            self.dataTable.setRowCount(df.shape[0])
            self.dataTable.setColumnCount(df.shape[1])

            # Set headers
            headers = df.columns.tolist()
            self.dataTable.setHorizontalHeaderLabels(headers)

            # Populate table with data
            for row in range(df.shape[0]):
                for col in range(df.shape[1]):
                    self.dataTable.setItem(row, col, QtWidgets.QTableWidgetItem(str(df.iat[row, col])))

        except ValueError as ve:
            QtWidgets.QMessageBox.critical(self, "Error", f"File format error: {ve}")
        except FileNotFoundError:
            QtWidgets.QMessageBox.critical(self, "Error", "File not found.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def on_menu_button_click(self, page_name):
        # Reset all buttons
        self.set_button_styles(self.btnHome, 'home', ":/images/images/icons/icons8-home-50.png")
        self.set_button_styles(self.btnData, 'data', ":/images/images/icons/icons8-table-50.png")
        self.set_button_styles(self.btnModel, 'model', ":/images/images/icons/icons8-machine-learning-50.png")
        self.set_button_styles(self.btnHelp, 'help', ":/images/images/icons/icons8-help-50.png")

        # Set active button style
        icons = {
            'home': ":/images/images/icons/icons8-home-30.png",
            'data': ":/images/images/icons/icons8-table-30.png",
            'model': ":/images/images/icons/icons8-machine-learning-30.png",
            'help': ":/images/images/icons/icons8-help-30.png",
        }

        buttons = {
            'home': self.btnHome,
            'data': self.btnData,
            'model': self.btnModel,
            'help': self.btnHelp,
        }

        if page_name in buttons:
            button = buttons[page_name]
            button.setStyleSheet(get_active_button_stylesheet())
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(icons[page_name]), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            button.setIcon(icon)
            button.setIconSize(QtCore.QSize(30, 30))

        # Show the corresponding page in the stacked widget
        page_index = {
            'home': 0,
            'data': 1,
            'model': 3,
            'help': 4
        }
        self.stackedWidget.setCurrentIndex(page_index[page_name])
    def set_button_styles(self,button, page_name, icon_path):
        # Set button styles using the given stylesheet
        button.setStyleSheet(get_button_stylesheet())

        # Only update icon if it's not already set in the active state
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        button.setIcon(icon)
        button.setIconSize(QtCore.QSize(30, 30))
    

# Main application
app = QtWidgets.QApplication(sys.argv)

# Show the login screen
login_screen = MainScreen()
login_screen.show()

sys.exit(app.exec_())
