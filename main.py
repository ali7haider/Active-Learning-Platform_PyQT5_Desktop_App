import sys
from PyQt5 import QtWidgets, uic, QtGui, QtCore
import resources_rc
from stylesheet_helper import get_button_stylesheet, get_active_button_stylesheet
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QLayout,QLineEdit

class MainScreen(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainScreen, self).__init__()
        uic.loadUi('main.ui', self)  # Load the UI file
        self.stackedWidget.setCurrentIndex(0)
        self.file_path = None  # Variable to store the file path
        self.df=None
        self.column_bounds=None


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

        # Connect buttons to their handlers
        self.btnSingleObj.clicked.connect(self.handle_single_obj_click)
        self.btnMultiObj.clicked.connect(self.handle_multi_obj_click)
        self.btnSubmitSingleObj.clicked.connect(self.collect_column_bounds)
    def get_qframe_in_layout(self,layout):
        item = layout.itemAt(1)
        if item:
            return item.widget() 

        return None
    def get_first_two_line_edits_from_frame(self,frame):
        if not frame or not frame.layout():
            return None, None

        line_edits = []
        for i in range(frame.layout().count()):
            item = frame.layout().itemAt(i)
            if isinstance(item.widget(), QLineEdit):
                line_edits.append(item.widget())
                if len(line_edits) == 2:
                    break

        return line_edits[0] if line_edits else None, line_edits[1] if len(line_edits) > 1 else None
    def collect_column_bounds(self):
        try:
            # Dictionary to store column bounds
            bounds_dict = {}

            # Loop through each row in the frameSearchSpace layout
            layout_count = self.frameSearchSpace.layout().count()
            for i in range(layout_count):
                layout = self.frameSearchSpace.layout().itemAt(i).layout()
                
                if layout is not None:  # Ensure layout is not None
                    # Get column name from label widget
                    if isinstance(layout, QLayout):
                        input_frame_layout = self.get_qframe_in_layout(layout)
                    column_label_widget = layout.itemAt(0).widget()
                    column_name = column_label_widget.text() if column_label_widget else None

                    # Check if the layout contains the input frame (itemAt(1))
                    if input_frame_layout:
                        # Get lower bound input widget from the layout
                        first_line_edit, second_line_edit = self.get_first_two_line_edits_from_frame(input_frame_layout)
                        lower_bound = first_line_edit.text() if first_line_edit.text() else 0
                        upper_bound = second_line_edit.text() if second_line_edit.text() else 0 

                        # Save column name and its bounds into the dictionary
                        if column_name:
                            bounds_dict[column_name] = {'lower': lower_bound, 'upper': upper_bound}
                    else:
                        print(f"Input frame layout at index {i} is None.")
            self.column_bounds = bounds_dict
            print(self.column_bounds)   
            # Change stack widget page after collecting column bounds
            self.stackedWidget.setCurrentIndex(5)  # Assuming the next page is indexed at 1

        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"An error occurred: {e}")


            
    def show_columns_for_bounds(self):
        try:
           # Clear existing layouts (if any)
            while self.frameSearchSpace.layout().count() > 0:
                item = self.frameSearchSpace.layout().takeAt(0)
                if item.layout():
                    if isinstance(item, QLayout): 
                        # Remove input_frame_layout if it exists
                        input_frame_layout = self.get_qframe_in_layout(item) 
                        if input_frame_layout:
                            input_frame_layout.deleteLater() 
                    item.layout().deleteLater() 
                elif item.widget():
                    item.widget().deleteLater()
            # Update the frame to reflect the changes
            self.frameSearchSpace.update() 

            # Check if the DataFrame is loaded
            if not hasattr(self, 'df') or self.df is None:
                QtWidgets.QMessageBox.warning(None, "Warning", "No data loaded to display columns.")
                return

            # Get the column names
            column_names = self.df.columns

            # Loop through columns and create a row for each column
            for column in column_names:
                row_layout = QtWidgets.QHBoxLayout()

                # Add column name label
                column_label = QtWidgets.QLabel(column, self.frameSearchSpace)
                column_label.setMinimumHeight(30)  # Set minimum height for the label
                row_layout.addWidget(column_label)

                # Create a frame for inputs (Lower Bound and Upper Bound)
                input_frame = QtWidgets.QFrame(self.frameSearchSpace)
                input_frame_layout = QtWidgets.QHBoxLayout(input_frame)
                input_frame_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
                input_frame_layout.setSpacing(50)  # Spacing between inputs

                # Add lower bound input to the frame
                lower_bound_input = QtWidgets.QLineEdit(input_frame)
                lower_bound_input.setPlaceholderText("Lower Bound")
                lower_bound_input.setMinimumHeight(30)  # Set minimum height
                input_frame_layout.addWidget(lower_bound_input)

                # Add upper bound input to the frame
                upper_bound_input = QtWidgets.QLineEdit(input_frame)
                upper_bound_input.setPlaceholderText("Upper Bound")
                upper_bound_input.setMinimumHeight(30)  # Set minimum height
                input_frame_layout.addWidget(upper_bound_input)

                # Align the input frame to the right
                input_frame.setLayout(input_frame_layout)
                row_layout.addWidget(input_frame, alignment=QtCore.Qt.AlignRight)

                # Add the row to the frame's layout
                self.frameSearchSpace.layout().addLayout(row_layout)

        except AttributeError as ae:
            QtWidgets.QMessageBox.critical(None, "Error", f"Attribute error occurred: {ae}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"An unexpected error occurred: {e}")


    def handle_single_obj_click(self):
        if not self.file_path:
            # Show message box if file path does not exist
            QtWidgets.QMessageBox.warning(None, "Warning", "Please select a file in the Data page.")
            return
        # Show columns for bounds
        self.show_columns_for_bounds()
        self.stackedWidget.setCurrentIndex(4)
    

    def handle_multi_obj_click(self):
        if not self.file_path:
            # Show message box if file path does not exist
            QtWidgets.QMessageBox.warning(None, "Warning", "Please select a file in the Data page.")
            return
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
                self.df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")

            # Save the file path and enable the button
            self.file_path = file_path
            self.dataTable.setRowCount( self.df.shape[0])
            self.dataTable.setColumnCount( self.df.shape[1])

            # Set headers
            headers =  self.df.columns.tolist()
            self.dataTable.setHorizontalHeaderLabels(headers)

            # Populate table with data
            for row in range( self.df.shape[0]):
                for col in range( self.df.shape[1]):
                    item = QtWidgets.QTableWidgetItem(str(self.df.iat[row, col]))
                    item.setTextAlignment(QtCore.Qt.AlignCenter)  # Center-align text
                    self.dataTable.setItem(row, col, item)

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
