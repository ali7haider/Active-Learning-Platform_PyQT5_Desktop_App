import sys
from PyQt5 import QtWidgets, uic, QtGui, QtCore
import resources_rc
from stylesheet_helper import get_button_stylesheet, get_active_button_stylesheet
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QLayout,QLineEdit
import subprocess
import threading
import json     
import os
from objective_files.test_single import run_optimization
class MainScreen(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainScreen, self).__init__()
        uic.loadUi('main.ui', self)  # Load the UI file
        self.stackedWidget.setCurrentIndex(0)  # Set initial page to index 0
        self.file_path = None  # Variable to store the file path
        self.df = None  # DataFrame variable
        self.column_bounds = None  # Bounds for columns

        # Connect browse button to the function
        self.btnBrowse.clicked.connect(self.open_file_dialog)  # Open file dialog when clicked

        # Connect menu buttons to their functions
        self.btnHome.clicked.connect(lambda: self.on_menu_button_click('home'))  # Navigate to home page
        self.btnData.clicked.connect(lambda: self.on_menu_button_click('data'))  # Navigate to data page
        self.btnModel.clicked.connect(lambda: self.on_menu_button_click('model'))  # Navigate to model page
        self.btnHelp.clicked.connect(lambda: self.on_menu_button_click('help'))  # Navigate to help page
        self.btnNext.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))  # Move to the next page

        # Set stylesheet for home button
        self.btnHome.setStyleSheet(get_active_button_stylesheet())
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(':/images/images/icons/icons8-home-30.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnHome.setIcon(icon)
        self.btnHome.setIconSize(QtCore.QSize(30, 30))

        # Adjust column widths of the data table
        self.dataTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        # Connect buttons to their handlers
        self.btnSingleObj.clicked.connect(self.handle_single_obj_click)  # Handle single objective button click
        self.btnMultiObj.clicked.connect(self.handle_multi_obj_click)  # Handle multi-objective button click
        self.btnSubmitSingleObj.clicked.connect(self.collect_column_bounds)  # Collect column bounds
        self.btnRunExperiment.clicked.connect(self.btnRunExperiment_click)  # Run the experiment

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
            # Change stack widget page after collecting column bounds
            self.change_stack_widget_page_and_fill_combo()
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"An error occurred: {e}")


    def change_stack_widget_page_and_fill_combo(self):
        try:
            # Change the stack widget to the desired page (assuming page 5)
            self.stackedWidget.setCurrentIndex(5)

            # Check if the DataFrame is loaded
            if not hasattr(self, 'df') or self.df is None:
                QtWidgets.QMessageBox.warning(None, "Warning", "No data loaded to fill the combo box.")
                return

            # Get the column names from the DataFrame
            column_names = self.df.columns

            # Fill the combo box with these column names
            self.cmbxTarget.clear()  # Clear existing items first
            self.cmbxTarget.addItems(column_names)

        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"An error occurred: {e}")        
        
    def btnRunExperiment_click(self):
        try:
            # Check if the DataFrame is loaded
            if not hasattr(self, 'df') or self.df is None:
                QtWidgets.QMessageBox.warning(None, "Warning", "No data loaded to display columns.")
                return
            # Get the values from the input fields and combo box
            batch_size = self.txtBatchSize.text().strip()
            n_obj = self.txtNObj.text().strip()
            n_var = self.txtNVar.text().strip()
            target_column = self.cmbxTarget.currentText().strip()

            # Check for null or empty values
            if not batch_size or not n_obj or not n_var or not target_column:
                QtWidgets.QMessageBox.warning(None, "Warning", "Please fill all fields before running the experiment.")
                return

            # Convert numeric inputs to integers if possible
            try:
                batch_size = int(batch_size)
                n_obj = int(n_obj)
                n_var = int(n_var)
            except ValueError:
                QtWidgets.QMessageBox.critical(None, "Error", "Invalid numeric input. Please enter valid integers.")
                return
            self.stackedWidget.setCurrentIndex(6)           
            # Start the experiment in a separate thread
            self.experiment_thread = ExperimentThread(batch_size, n_obj, n_var, target_column, self.df, self.column_bounds)
            self.experiment_thread.finished.connect(self.handle_experiment_result)
            self.experiment_thread.start()
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"An error occurred: {e}")

    def handle_experiment_result(self, result):
        if isinstance(result, pd.DataFrame):
            print('Result',result)  # Update UI or display results here
        else:
            QtWidgets.QMessageBox.critical(None, "Error", result)
    
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
                    column_label_widget = item.itemAt(0).widget()
                    column_label_widget.deleteLater() 
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
    

class ExperimentThread(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)  # Signal to emit the result

    def __init__(self, batch_size, n_obj, n_var, target_column, df, column_bounds):
        super().__init__()
        self.batch_size = batch_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.target_column = target_column
        self.df = df
        self.column_bounds = column_bounds

    def run(self):
        try:
            # Run the experiment in the background
            result = self.run_experiment(
                self.batch_size, self.n_obj, self.n_var, self.target_column, self.df, self.column_bounds
            )
            self.finished.emit(result)  # Emit the result when done
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")
    def run_experiment(self, batch_size, n_obj, n_var, target_column, df, column_bounds):
        try:
            # Parse the column bounds into lower and upper bounds, excluding the target column
            lower_bounds = [column_bounds[col]['lower'] for col in column_bounds.keys() if col != target_column]
            upper_bounds = [column_bounds[col]['upper'] for col in column_bounds.keys() if col != target_column]
            # Ensure bounds are numeric
            lower_bounds = [float(b) for b in lower_bounds]  # Convert lower bounds to float
            upper_bounds = [float(b) for b in upper_bounds]  # Convert upper bounds to float

            # Check if the DataFrame contains the required columns
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' is not in the DataFrame.")

            # Call the optimization function
            df_candidates = run_optimization(
                df=df,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                n_var=n_var,
                n_obj=n_obj,
               target_column=target_column,    
                batch_size=batch_size
            )

            return df_candidates
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Failed to run experiment: {e}")


# Main application
app = QtWidgets.QApplication(sys.argv)

# Show the login screen
login_screen = MainScreen()
login_screen.show()

sys.exit(app.exec_())
