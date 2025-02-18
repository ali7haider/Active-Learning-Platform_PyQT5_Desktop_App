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
from objective_files.single_objective_code_func import run_optimization
from objective_files.multi_objective_code_two_func import  run_multi_objective_optimization_two
from objective_files.multi_objective_code_three_func import run_multi_objective_optimization_three

from datetime import datetime
import csv

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
        self.btnRunExperiment.clicked.connect(self.btnRunExperiment_click)  # Run the experiment

        self.btnRunExperimentMultiObj.clicked.connect(self.btnRunMultiObjExperiment_click)  # Run the experiment
        self.btnDownload.clicked.connect(self.btnDownload_click)
        self.btnTargetTwo.clicked.connect(lambda: self.update_target_visibility(2))
        self.btnTargetThree.clicked.connect(lambda: self.update_target_visibility(3))

        self.btnSubmitSingleObj.clicked.connect(lambda: self.collect_column_bounds(self.frameSearchSpace))
        self.btnSubmitMultiObj.clicked.connect(lambda: self.collect_column_bounds(self.frameSearchSpace_2))


    def handle_multi_obj_click(self):
        if not self.file_path:
            # Show message box if file path does not exist
            QtWidgets.QMessageBox.warning(None, "Warning", "Please select a file in the Data page.")
            return
        
        # change the stack widget page to 4 
        self.stackedWidget.setCurrentIndex(8)
        # # Connect buttons to update target visibility
        # self.btnTargetTwo.clicked.connect(lambda: self.update_target_visibility(2))
        # self.btnTargetThree.clicked.connect(lambda: self.update_target_visibility(3))

    def btnRunMultiObjExperiment_click(self):
        try:
            # Check if the DataFrame is loaded
            if not hasattr(self, 'df') or self.df is None:
                QtWidgets.QMessageBox.warning(None, "Warning", "No data loaded to display columns.")
                return

            # Get input values
            batch_size = self.txtBatchSizeMulti.text().strip()
            n_var = self.txtNVarMulti.text().strip()
            target1 = self.cmbxTarget1.currentText().strip()
            ref1 = self.txtReference1.text().strip()
            target2 = self.cmbxTarget2.currentText().strip()
            ref2 = self.txtReference2.text().strip()

            # Ensure mandatory fields are filled
            if not batch_size or not n_var or not target1 or not ref1 or not target2 or not ref2:
                QtWidgets.QMessageBox.warning(None, "Warning", "Please fill all required fields before running the experiment.")
                return

            # Convert batch size & variable count to integers
            try:
                batch_size = int(batch_size)
                n_var = int(n_var)
            except ValueError:
                QtWidgets.QMessageBox.critical(None, "Error", "Invalid numeric input. Please enter valid integers.")
                return

            # Store mandatory targets
            targets = [(target1, ref1), (target2, ref2)]

            # Check third target (optional) - included only if ref3 is a valid number and visible
            if self.cmbxTarget3.isVisible() and self.txtReference3.isVisible():
                target3 = self.cmbxTarget3.currentText().strip()
                ref3 = self.txtReference3.text().strip()
                try:
                    if ref3:  # If ref3 is not empty, check if it's a valid number
                        float(ref3)  # Validate number
                        targets.append((target3, ref3))
                except ValueError:
                    pass  # Ignore target3 if ref3 is invalid

            # Show loading message
            self.lblLoading_2.setText("Loading... Please Wait")

            # Update UI: Reset results table & change stack widget page
            # Start experiment in a separate thread
            # self.experiment_thread = MultiObjExperimentThread(batch_size, n_var, targets, self.df, self.column_bounds)
            # self.experiment_thread.finished.connect(self.handle_experiment_result)
            # self.experiment_thread.start()
            # Show the DataFrame in the table
            self.resultTable_2.setRowCount(0)  # Set rows
            self.resultTable_2.setColumnCount(1)  # Set columns
            self.stackedWidget.setCurrentIndex(11) 
            self.experiment_thread_multi = MultiObjExperimentThread(
            batch_size=batch_size,
            n_var=n_var,
            target_columns=targets,
            df=self.df,
            column_bounds=self.column_bounds
        )
        
            # Connect the finished signal to a handler
            self.experiment_thread_multi.finished.connect(self.handle_experiment_result_multi)  # Make sure to define this function
            self.experiment_thread_multi.start()

        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"An error occurred: {e}")


    def update_target_visibility(self, target_count):
        if target_count == 2:
            self.lblTarget3.hide()
            self.cmbxTarget3.hide()
            self.txtReference3.hide()
        elif target_count == 3:
            self.lblTarget3.show()
            self.cmbxTarget3.show()
            self.txtReference3.show()

        self.show_columns_for_bounds(self.frameSearchSpace_2)
        self.stackedWidget.setCurrentIndex(9)  # Move to page 9 after selection

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
    def collect_column_bounds(self, frame):
        try:
            # Dictionary to store column bounds
            bounds_dict = {}

            # Ensure the frame has a layout
            if frame.layout() is None:
                QtWidgets.QMessageBox.warning(None, "Warning", "No layout found in the frame.")
                return

            # Loop through each row in the provided frame layout
            layout_count = frame.layout().count()
            for i in range(layout_count):
                layout = frame.layout().itemAt(i).layout()

                if layout is not None:  # Ensure layout exists
                    # Get column name from label widget
                    column_label_widget = layout.itemAt(0).widget()
                    column_name = column_label_widget.text() if column_label_widget else None

                    # Get the input frame layout (itemAt(1) should contain the input fields)
                    input_frame = layout.itemAt(1).widget() if layout.count() > 1 else None
                    if input_frame and isinstance(input_frame, QtWidgets.QFrame):
                        input_frame_layout = input_frame.layout()

                        # Retrieve lower and upper bound input fields
                        if input_frame_layout:
                            first_line_edit, second_line_edit = self.get_first_two_line_edits_from_frame(input_frame_layout)
                            lower_bound = first_line_edit.text() if first_line_edit and first_line_edit.text() else "0"
                            upper_bound = second_line_edit.text() if second_line_edit and second_line_edit.text() else "0"

                            # Save column name and its bounds into the dictionary
                            if column_name:
                                bounds_dict[column_name] = {'lower': lower_bound, 'upper': upper_bound}
                    else:
                        print(f"Input frame layout at index {i} is missing.")

            self.column_bounds = bounds_dict
            # Change stack widget page after collecting column bounds
            if frame == self.frameSearchSpace:
                self.change_stack_widget_page_and_fill_combo(5)
            elif frame == self.frameSearchSpace_2:
                self.change_stack_widget_page_and_fill_combo(10)

        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"An error occurred: {e}")


    def change_stack_widget_page_and_fill_combo(self,index):
        try:
            # Change the stack widget to the desired page (assuming page 5)
            self.stackedWidget.setCurrentIndex(index)

            # Check if the DataFrame is loaded
            if not hasattr(self, 'df') or self.df is None:
                QtWidgets.QMessageBox.warning(None, "Warning", "No data loaded to fill the combo box.")
                return

            # Get the column names from the DataFrame
            column_names = self.df.columns

            # Fill the combo box with these column names
            self.cmbxTarget.clear()  # Clear existing items first
            self.cmbxTarget.addItems(column_names)
            self.cmbxTarget1.clear()
            self.cmbxTarget1.addItems(column_names)
            self.cmbxTarget2.clear()
            self.cmbxTarget2.addItems(column_names)
            self.cmbxTarget3.clear()
            self.cmbxTarget3.addItems(column_names)

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
            self.lblLoading.setText("Loading...Please Wait")
            # Show the DataFrame in the table
            self.resultTable.setRowCount(0)  # Set rows
            self.resultTable.setColumnCount(1)  # Set columns
            self.stackedWidget.setCurrentIndex(6)           
            # Start the experiment in a separate thread
            self.experiment_thread = ExperimentThread(batch_size, n_obj, n_var, target_column, self.df, self.column_bounds)
            self.experiment_thread.finished.connect(self.handle_experiment_result)
            self.experiment_thread.start()
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"An error occurred: {e}")

    def handle_experiment_result(self, result):
        try:
            if isinstance(result, pd.DataFrame):
                # Update the label
                self.lblLoading.setText("Data shown successfully")

                # Show the DataFrame in the table
                self.resultTable.setRowCount(result.shape[0])  # Set rows
                self.resultTable.setColumnCount(result.shape[1])  # Set columns
                self.resultTable.setHorizontalHeaderLabels(result.columns)  # Set column headers

                # Populate the table with DataFrame values
                for row in range(result.shape[0]):
                    for col in range(result.shape[1]):
                        item = QtWidgets.QTableWidgetItem(str(result.iat[row, col]))
                        item.setTextAlignment(QtCore.Qt.AlignCenter)  # Center-align text
                        self.resultTable.setItem(row, col, item)

                # Resize the table columns to fit contents
                self.resultTable.resizeColumnsToContents()
                self.resultTable.resizeRowsToContents()
                self.resultTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

            else:
                # Show error message if result is not a DataFrame
                QtWidgets.QMessageBox.critical(None, "Error", result)

        except Exception as e:
            # Handle unexpected errors
            QtWidgets.QMessageBox.critical(None, "Critical Error", f"An error occurred while displaying results: {e}")

    def handle_experiment_result_multi(self, result):
        try:
            if isinstance(result, pd.DataFrame):
                # Update the label
                self.lblLoading_2.setText("Data shown successfully")

                # Show the DataFrame in the table
                self.resultTable_2.setRowCount(result.shape[0])  # Set rows
                self.resultTable_2.setColumnCount(result.shape[1])  # Set columns
                self.resultTable_2.setHorizontalHeaderLabels(result.columns)  # Set column headers

                # Populate the table with DataFrame values
                for row in range(result.shape[0]):
                    for col in range(result.shape[1]):
                        item = QtWidgets.QTableWidgetItem(str(result.iat[row, col]))
                        item.setTextAlignment(QtCore.Qt.AlignCenter)  # Center-align text
                        self.resultTable_2.setItem(row, col, item)

                # Resize the table columns to fit contents
                self.resultTable_2.resizeColumnsToContents()
                self.resultTable_2.resizeRowsToContents()
                self.resultTable_2.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

            else:
                # Show error message if result is not a DataFrame
                QtWidgets.QMessageBox.critical(None, "Error", result)

        except Exception as e:
            # Handle unexpected errors
            QtWidgets.QMessageBox.critical(None, "Critical Error", f"An error occurred while displaying results: {e}")


    def btnDownload_click(self):
        try:
            # Get current datetime for the file name
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"results_{current_time}.csv"

            # Open a file dialog to select the save location
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                None,
                "Save Results",
                file_name,
                "CSV Files (*.csv)"
            )

            if not file_path:  # If no file is selected, return
                return

            # Open the file for writing
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                # Write the headers
                headers = [self.resultTable.horizontalHeaderItem(i).text() for i in range(self.resultTable.columnCount())]
                writer.writerow(headers)

                # Write table rows
                for row in range(self.resultTable.rowCount()):
                    row_data = [
                        self.resultTable.item(row, col).text() if self.resultTable.item(row, col) else ""
                        for col in range(self.resultTable.columnCount())
                    ]
                    writer.writerow(row_data)

            # Show success message
            QtWidgets.QMessageBox.information(None, "Success", f"Results saved successfully as {file_name}.")

        except Exception as e:
            # Show error message in case of failure
            QtWidgets.QMessageBox.critical(None, "Error", f"Failed to save results: {e}")
    
    
    def show_columns_for_bounds(self, frame):
        try:
            # Ensure the frame has a layout
            if frame.layout() is None:
                frame.setLayout(QtWidgets.QVBoxLayout())

            # Clear existing layouts (if any)
            while frame.layout().count() > 0:
                item = frame.layout().takeAt(0)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()

            # Update the frame to reflect changes
            frame.update()

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
                column_label = QtWidgets.QLabel(column, frame)
                column_label.setMinimumHeight(30)
                row_layout.addWidget(column_label)

                # Create a frame for inputs (Lower Bound and Upper Bound)
                input_frame = QtWidgets.QFrame(frame)
                input_frame_layout = QtWidgets.QHBoxLayout(input_frame)
                input_frame_layout.setContentsMargins(0, 0, 0, 0)
                input_frame_layout.setSpacing(50)

                # Add lower bound input
                lower_bound_input = QtWidgets.QLineEdit(input_frame)
                lower_bound_input.setPlaceholderText("Lower Bound")
                lower_bound_input.setMinimumHeight(30)
                input_frame_layout.addWidget(lower_bound_input)

                # Add upper bound input
                upper_bound_input = QtWidgets.QLineEdit(input_frame)
                upper_bound_input.setPlaceholderText("Upper Bound")
                upper_bound_input.setMinimumHeight(30)
                input_frame_layout.addWidget(upper_bound_input)

                # Align input frame to the right
                input_frame.setLayout(input_frame_layout)
                row_layout.addWidget(input_frame, alignment=QtCore.Qt.AlignRight)

                # Add the row layout to the main frame's layout
                frame.layout().addLayout(row_layout)

        except AttributeError as ae:
            QtWidgets.QMessageBox.critical(None, "Error", f"Attribute error occurred: {ae}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"An unexpected error occurred: {e}")



    def handle_single_obj_click(self):
        if not self.file_path:
            # Show message box if file path does not exist
            QtWidgets.QMessageBox.warning(None, "Warning", "Please select a file in the Data page.")
            return
        # Disable the txtNObj field and set its text to "1"
        self.txtNObj.setText("1")
        self.txtNObj.setDisabled(True)
        # Show columns for bounds
        self.show_columns_for_bounds(self.frameSearchSpace)
        self.stackedWidget.setCurrentIndex(4)

    

    
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
            'help': 7
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
class MultiObjExperimentThread(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)  # Signal to emit the result

    def __init__(self, batch_size, n_var, target_columns, df, column_bounds):
        super().__init__()
        self.batch_size = batch_size
        self.n_var = n_var
        self.target_columns = target_columns  # List of target columns for multi-objective
        self.df = df
        self.column_bounds = column_bounds

    def run(self):
        try:
            
            # Run the experiment in the background
            result = self.run_experiment(
                self.batch_size, self.n_var, self.target_columns, self.df, self.column_bounds
            )
            
            self.finished.emit(result)  # Emit the result when done
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

    def run_experiment(self, batch_size, n_var, target_columns, df, column_bounds):
        try:

            # Extract column names and reference numbers from target_columns
            target_column_names = [col[0] for col in target_columns]  # Extract only column names
            reference_numbers = [col[1] for col in target_columns]  

            
            # Ensure that target_column_names are strings
            if not all(isinstance(col, str) for col in target_column_names):
                raise ValueError("[ERROR] One or more target column names are not strings!")

            # Extract non-target columns safely
            non_target_columns = [col for col in column_bounds.keys() if col not in target_column_names]

            # Parse lower and upper bounds
            lower_bounds = [column_bounds[col]['lower'] for col in non_target_columns]
            upper_bounds = [column_bounds[col]['upper'] for col in non_target_columns]


            # Convert bounds to int for safety
            lower_bounds = [int(b) for b in lower_bounds]  
            upper_bounds = [int(b) for b in upper_bounds]

            
            # Validate that target columns exist in the DataFrame
            if not all(target_column in df.columns for target_column in target_column_names):
                raise ValueError("[ERROR] One or more target columns are missing in the DataFrame!")

            # Determine the number of objectives
            n_obj = len(target_columns)

            # Call optimization function based on objectives
            if n_obj == 2:
                print("[DEBUG] Running two-objective optimization...")
                df_candidates = run_multi_objective_optimization_two(
                    df=df,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    n_var=n_var,
                    target_columns=target_column_names,    
                    batch_size=batch_size,
                    ref_point=reference_numbers
                )
            elif n_obj == 3:
                print("[DEBUG] Running three-objective optimization...")
                df_candidates = run_multi_objective_optimization_three(
                    df=df,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    n_var=n_var,
                    target_columns=target_column_names,    
                    batch_size=batch_size,
                    ref_point=reference_numbers
                )
            else:
                raise ValueError("[ERROR] Currently only supports 2 or 3 objectives for multi-objective optimization.")

            return df_candidates
        except Exception as e:
            print(f"[ERROR] Failed to run experiment: {e}")
            QtWidgets.QMessageBox.critical(None, "Error", f"Failed to run experiment: {e}")

# Main application
app = QtWidgets.QApplication(sys.argv)

# Show the login screen
login_screen = MainScreen()
login_screen.show()

sys.exit(app.exec_())
