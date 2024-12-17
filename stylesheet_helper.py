from PyQt5 import QtWidgets, uic, QtGui, QtCore

def get_button_stylesheet():
    return '''
        QPushButton {
            color: white;                      /* Text color */
            border-radius: 5px;                /* Rounded corners */
            padding: 6px 12px;                 /* Padding inside button */
            font-weight: bold;                 /* Bold text */
            background-color:#068fff;
            font:  15pt "Calibri";
            padding-left: 10px; /* Add padding for spacing */
            padding-right: 0px; /* Remove right padding to avoid extra space */
            text-align: left; /* Align text and image to the left */
        }
        QPushButton:hover {
            background-color: #067EE0;  
            color:#fff;       /* Slightly darker color on hover */
        }
        QPushButton:pressed {
            padding-left: 14px;                /* Slight movement for press effect */
            padding-top: 6px;                  /* Slight movement for press effect */
        }
    '''

def get_active_button_stylesheet():
    return '''
        QPushButton {
            background-color: white;
            color: #068fff;
        }
    '''

