# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1194, 765)
        self.styleSheet = QtWidgets.QWidget(MainWindow)
        self.styleSheet.setStyleSheet("\n"
"QWidget{\n"
"    color: black;\n"
"    font: 75 11pt \"Calibri\";\n"
"\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Tooltip */\n"
"QToolTip {\n"
"    color: #ffffff;\n"
"    background-color: rgba(33, 37, 43, 180);\n"
"    border: 1px solid rgb(44, 49, 58);\n"
"    background-image: none;\n"
"    background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"    border: none;\n"
"    border-left: 2px solid rgb(255, 121, 198);\n"
"    text-align: left;\n"
"    padding-left: 8px;\n"
"    margin: 0px;\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Bg App */\n"
"\n"
"\n"
"\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"QTableWidget */\n"
"QTableWidget {    \n"
"    background-color: #444;\n"
"    padding: 5px;\n"
"    border-radius: 8px;\n"
"    gridline-color: #555;\n"
"    color: black;\n"
"    font: 11pt \"Calibri\";\n"
"}\n"
"QTableWidget::item {\n"
"    border-color: none;\n"
"    padding-left: 5px;\n"
"    padding-right: 5px;\n"
"    gridline-color: #555;\n"
"    border-bottom: 1px solid #555;\n"
"}\n"
"QTableWidget::item:selected {\n"
"    background-color: #00ABE8;\n"
"    color: #FFF;\n"
"}\n"
"QHeaderView::section {\n"
"    background-color:#068fff;\n"
"    max-width: 30px;\n"
"    border: 1px solid #BBB;\n"
"    border-style: none;\n"
"    border-bottom: 1px solid #BBB;\n"
"    border-right: 1px solid #BBB;\n"
"}\n"
"QTableWidget::horizontalHeader {    \n"
"    background-color: rgb(189, 147, 249);\n"
"    color: #FFF;\n"
"\n"
"}\n"
"QTableWidget QTableCornerButton::section {\n"
"    border: none;\n"
"    background-color: #068fff;\n"
"    padding: 3px;\n"
"    border-top-left-radius: 8px;\n"
"}\n"
"QHeaderView::section:horizontal {\n"
"    border: none;\n"
"    background-color: #068fff;\n"
"    padding: 3px;\n"
"    color: #FFF;\n"
"    font: 75 12pt \"Calibri\";\n"
"\n"
"}\n"
"QHeaderView::section:vertical {\n"
"    border: none;\n"
"    background-color: #fff;\n"
"    padding: 5px;\n"
"    border-bottom: 2px solid #555;\n"
"    border-left-bottom: 2px solid #555;\n"
"\n"
"}\n"
"QScrollBar:horizontal {\n"
"    background: #FFF;\n"
"    height: 8px;\n"
"    margin: 0px 16px 0 16px;\n"
"    border: none;\n"
"}\n"
"QScrollBar:vertical {\n"
"    background: #FFF;\n"
"    width: 8px;\n"
"    margin: 16px 0 16px 0;\n"
"    border: none;\n"
"}\n"
"QScrollBar::handle:horizontal {\n"
"    background: #3333;\n"
"    min-width: 20px;\n"
"    border-radius: 4px;\n"
"}\n"
"QScrollBar::handle:vertical {\n"
"    background: #068fff; /* Changed scrollbar color to #068fff */\n"
"    min-height: 20px;\n"
"    border-radius: 4px;\n"
"}\n"
"QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,\n"
"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {\n"
"    background: none;\n"
"}\n"
"QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal,\n"
"QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {\n"
"    background: none;\n"
"}\n"
"\n"
"\n"
"\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"LineEdit */\n"
"QLineEdit {\n"
"background-color: rgb(33, 37, 43);\n"
"    border-radius: 5px;\n"
"    border: 2px solid rgb(33, 37, 43);\n"
"    padding-left: 10px;\n"
"    selection-color: rgb(255, 255, 255);\n"
"    selection-background-color: rgb(255, 121, 198);\n"
"}\n"
"QLineEdit:hover {\n"
"    border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QLineEdit:focus {\n"
"    border: 2px solid rgb(91, 101, 124);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"PlainTextEdit */\n"
"QPlainTextEdit {\n"
"    background-color: rgb(27, 29, 35);\n"
"    border-radius: 5px;\n"
"    padding: 10px;\n"
"    selection-color: rgb(255, 255, 255);\n"
"    selection-background-color: rgb(255, 121, 198);\n"
"}\n"
"QPlainTextEdit  QScrollBar:vertical {\n"
"    width: 8px;\n"
" }\n"
"QPlainTextEdit  QScrollBar:horizontal {\n"
"    height: 8px;\n"
" }\n"
"QPlainTextEdit:hover {\n"
"    border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QPlainTextEdit:focus {\n"
"    border: 2px solid rgb(91, 101, 124);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"ScrollBars */\n"
"QScrollBar:horizontal {\n"
"    border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    height: 8px;\n"
"    margin: 0px 21px 0 21px;\n"
"    border-radius: 0px;\n"
"}\n"
"QScrollBar::handle:horizontal {\n"
"    background:#068fff;;\n"
"    min-width: 25px;\n"
"    border-radius: 4px\n"
"}\n"
"QScrollBar::add-line:horizontal {\n"
"    border: none;\n"
"    background: rgb(55, 63, 77);\n"
"    width: 20px;\n"
"    border-top-right-radius: 4px;\n"
"    border-bottom-right-radius: 4px;\n"
"    subcontrol-position: right;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"QScrollBar::sub-line:horizontal {\n"
"    border: none;\n"
"    background: rgb(55, 63, 77);\n"
"    width: 20px;\n"
"    border-top-left-radius: 4px;\n"
"    border-bottom-left-radius: 4px;\n"
"    subcontrol-position: left;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal\n"
"{\n"
"     background: none;\n"
"}\n"
"QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal\n"
"{\n"
"     background: none;\n"
"}\n"
" QScrollBar:vertical {\n"
"    border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    width: 8px;\n"
"    margin: 21px 0 21px 0;\n"
"    border-radius: 0px;\n"
" }\n"
" QScrollBar::handle:vertical {    \n"
"    background: #068fff;;\n"
"    min-height: 25px;\n"
"    border-radius: 4px\n"
" }\n"
" QScrollBar::add-line:vertical {\n"
"     border: none;\n"
"    background: rgb(55, 63, 77);\n"
"     height: 20px;\n"
"    border-bottom-left-radius: 4px;\n"
"    border-bottom-right-radius: 4px;\n"
"     subcontrol-position: bottom;\n"
"     subcontrol-origin: margin;\n"
" }\n"
" QScrollBar::sub-line:vertical {\n"
"    border: none;\n"
"    background: rgb(55, 63, 77);\n"
"     height: 20px;\n"
"    border-top-left-radius: 4px;\n"
"    border-top-right-radius: 4px;\n"
"     subcontrol-position: top;\n"
"     subcontrol-origin: margin;\n"
" }\n"
" QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {\n"
"     background: none;\n"
" }\n"
"\n"
" QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {\n"
"     background: none;\n"
" }\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"CheckBox */\n"
"QCheckBox::indicator {\n"
"    border: 3px solid rgb(52, 59, 72);\n"
"    width: 15px;\n"
"    height: 15px;\n"
"    border-radius: 10px;\n"
"    background: rgb(44, 49, 60);\n"
"}\n"
"QCheckBox::indicator:hover {\n"
"    border: 3px solid rgb(58, 66, 81);\n"
"}\n"
"QCheckBox::indicator:checked {\n"
"    background: 3px solid rgb(52, 59, 72);\n"
"    border: 3px solid rgb(52, 59, 72);    \n"
"    background-image: url(:/icons/images/icons/cil-check-alt.png);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"RadioButton */\n"
"QRadioButton::indicator {\n"
"    border: 3px solid rgb(52, 59, 72);\n"
"    width: 15px;\n"
"    height: 15px;\n"
"    border-radius: 10px;\n"
"    background: rgb(44, 49, 60);\n"
"}\n"
"QRadioButton::indicator:hover {\n"
"    border: 3px solid rgb(58, 66, 81);\n"
"}\n"
"QRadioButton::indicator:checked {\n"
"    background: 3px solid rgb(94, 106, 130);\n"
"    border: 3px solid rgb(52, 59, 72);    \n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"ComboBox */\n"
"QComboBox{\n"
"    background-color: rgb(27, 29, 35);\n"
"    border-radius: 5px;\n"
"    border: 2px solid rgb(33, 37, 43);\n"
"    padding: 5px;\n"
"    padding-left: 10px;\n"
"}\n"
"QComboBox:hover{\n"
"    border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QComboBox::drop-down {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: top right;\n"
"    width: 25px; \n"
"    border-left-width: 3px;\n"
"    border-left-color: rgba(39, 44, 54, 150);\n"
"    border-left-style: solid;\n"
"    border-top-right-radius: 3px;\n"
"    border-bottom-right-radius: 3px;    \n"
"    background-image: url(:/icons/images/icons/cil-arrow-bottom.png);\n"
"    background-position: center;\n"
"    background-repeat: no-reperat;\n"
" }\n"
"QComboBox QAbstractItemView {\n"
"    color: rgb(255, 121, 198);    \n"
"    background-color: rgb(33, 37, 43);\n"
"    padding: 10px;\n"
"    selection-background-color: rgb(39, 44, 54);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Sliders */\n"
"QSlider::groove:horizontal {\n"
"    border-radius: 5px;\n"
"    height: 10px;\n"
"    margin: 0px;\n"
"    background-color: rgb(52, 59, 72);\n"
"}\n"
"QSlider::groove:horizontal:hover {\n"
"    background-color: rgb(55, 62, 76);\n"
"}\n"
"QSlider::handle:horizontal {\n"
"    background-color: rgb(189, 147, 249);\n"
"    border: none;\n"
"    height: 10px;\n"
"    width: 10px;\n"
"    margin: 0px;\n"
"    border-radius: 5px;\n"
"}\n"
"QSlider::handle:horizontal:hover {\n"
"    background-color: rgb(195, 155, 255);\n"
"}\n"
"QSlider::handle:horizontal:pressed {\n"
"    background-color: rgb(255, 121, 198);\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"    border-radius: 5px;\n"
"    width: 10px;\n"
"    margin: 0px;\n"
"    background-color: rgb(52, 59, 72);\n"
"}\n"
"QSlider::groove:vertical:hover {\n"
"    background-color: rgb(55, 62, 76);\n"
"}\n"
"QSlider::handle:vertical {\n"
"    background-color: rgb(189, 147, 249);\n"
"    border: none;\n"
"    height: 10px;\n"
"    width: 10px;\n"
"    margin: 0px;\n"
"    border-radius: 5px;\n"
"}\n"
"QSlider::handle:vertical:hover {\n"
"    background-color: rgb(195, 155, 255);\n"
"}\n"
"QSlider::handle:vertical:pressed {\n"
"    background-color: rgb(255, 121, 198);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"CommandLinkButton */\n"
"QCommandLinkButton {    \n"
"    color: rgb(255, 121, 198);\n"
"    border-radius: 5px;\n"
"    padding: 5px;\n"
"    color: rgb(255, 170, 255);\n"
"}\n"
"QCommandLinkButton:hover {    \n"
"    color: rgb(255, 170, 255);\n"
"    background-color: rgb(44, 49, 60);\n"
"}\n"
"QCommandLinkButton:pressed {    \n"
"    color: rgb(189, 147, 249);\n"
"    background-color: rgb(52, 58, 71);\n"
"}\n"
"\n"
"\n"
"\n"
"QLabel {\n"
"    font-weight: bold;   \n"
"}\n"
"\n"
"\n"
"\n"
"\n"
"#sideBar\n"
"{\n"
"    background-color: #279eff\n"
"}\n"
"\n"
"\n"
"")
        self.styleSheet.setObjectName("styleSheet")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.styleSheet)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.MainScreen = QtWidgets.QFrame(self.styleSheet)
        self.MainScreen.setStyleSheet("#loginFrame{\n"
"background-color:#EFEFEF;\n"
"}\n"
"QLabel\n"
"{\n"
"color:white;\n"
"font-style:Century Gothic;\n"
"font-size:12px;\n"
"}")
        self.MainScreen.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.MainScreen.setFrameShadow(QtWidgets.QFrame.Raised)
        self.MainScreen.setObjectName("MainScreen")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.MainScreen)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.sideBar = QtWidgets.QFrame(self.MainScreen)
        self.sideBar.setMinimumSize(QtCore.QSize(220, 0))
        self.sideBar.setMaximumSize(QtCore.QSize(220, 16777215))
        self.sideBar.setStyleSheet("\n"
"\n"
"QPushButton {\n"
"    color: white;                      /* Text color */\n"
"    border-radius: 5px;                /* Rounded corners */\n"
"    padding: 6px 12px;                 /* Padding inside button */\n"
"    font-weight: bold;                 /* Bold text */\n"
"    background-color:#068fff;\n"
"    font:  15pt \"Calibri\";\n"
"  padding-left: 10px; /* Add padding for spacing */\n"
"    padding-right: 0px; /* Remove right padding to avoid extra space */\n"
"    text-align: left; /* Align text and image to the left */\n"
"\n"
"\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: #ffffff;  \n"
"    color:#279eff;       /* Slightly darker color on hover */\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"       /* Darken the border further when pressed */\n"
"    padding-left: 14px;                /* Slight movement for press effect */\n"
"    padding-top: 6px;                  /* Slight movement for press effect */\n"
"}")
        self.sideBar.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.sideBar.setFrameShadow(QtWidgets.QFrame.Raised)
        self.sideBar.setObjectName("sideBar")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.sideBar)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(self.sideBar)
        self.frame.setMaximumSize(QtCore.QSize(16777215, 80))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_3.setContentsMargins(20, 0, 20, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/images/images/logo.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.verticalLayout.addWidget(self.frame)
        self.fameSideBarButtons = QtWidgets.QFrame(self.sideBar)
        self.fameSideBarButtons.setStyleSheet("")
        self.fameSideBarButtons.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.fameSideBarButtons.setFrameShadow(QtWidgets.QFrame.Raised)
        self.fameSideBarButtons.setObjectName("fameSideBarButtons")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.fameSideBarButtons)
        self.verticalLayout_2.setContentsMargins(7, 40, 7, -1)
        self.verticalLayout_2.setSpacing(30)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.btnHome = QtWidgets.QPushButton(self.fameSideBarButtons)
        self.btnHome.setMinimumSize(QtCore.QSize(0, 0))
        self.btnHome.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnHome.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btnHome.setStyleSheet("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/images/icons/icons8-home-50.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnHome.setIcon(icon)
        self.btnHome.setIconSize(QtCore.QSize(30, 30))
        self.btnHome.setObjectName("btnHome")
        self.verticalLayout_2.addWidget(self.btnHome)
        self.btnData = QtWidgets.QPushButton(self.fameSideBarButtons)
        self.btnData.setMinimumSize(QtCore.QSize(0, 50))
        self.btnData.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnData.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btnData.setAutoFillBackground(False)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/images/icons/icons8-table-50.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnData.setIcon(icon1)
        self.btnData.setIconSize(QtCore.QSize(30, 30))
        self.btnData.setObjectName("btnData")
        self.verticalLayout_2.addWidget(self.btnData)
        self.btnModel = QtWidgets.QPushButton(self.fameSideBarButtons)
        self.btnModel.setMinimumSize(QtCore.QSize(0, 50))
        self.btnModel.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/images/icons/icons8-machine-learning-50.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnModel.setIcon(icon2)
        self.btnModel.setIconSize(QtCore.QSize(30, 30))
        self.btnModel.setObjectName("btnModel")
        self.verticalLayout_2.addWidget(self.btnModel)
        self.btnHelp = QtWidgets.QPushButton(self.fameSideBarButtons)
        self.btnHelp.setMinimumSize(QtCore.QSize(0, 50))
        self.btnHelp.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/images/icons/icons8-help-50.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnHelp.setIcon(icon3)
        self.btnHelp.setIconSize(QtCore.QSize(30, 30))
        self.btnHelp.setObjectName("btnHelp")
        self.verticalLayout_2.addWidget(self.btnHelp)
        self.verticalLayout.addWidget(self.fameSideBarButtons, 0, QtCore.Qt.AlignTop)
        self.horizontalLayout_5.addWidget(self.sideBar)
        self.frame_5 = QtWidgets.QFrame(self.MainScreen)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.stackedWidget = QtWidgets.QStackedWidget(self.frame_5)
        self.stackedWidget.setStyleSheet("background-color:white;")
        self.stackedWidget.setObjectName("stackedWidget")
        self.homePage = QtWidgets.QWidget()
        self.homePage.setStyleSheet("QLabel{\n"
"    font:  12pt \"Calibri\";\n"
"color:black;\n"
"}")
        self.homePage.setObjectName("homePage")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.homePage)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_2 = QtWidgets.QFrame(self.homePage)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_3 = QtWidgets.QLabel(self.frame_2)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_4.addWidget(self.label_3)
        self.verticalLayout_3.addWidget(self.frame_2)
        self.stackedWidget.addWidget(self.homePage)
        self.DataPage = QtWidgets.QWidget()
        self.DataPage.setStyleSheet("QLabel{\n"
"    font:  12pt \"Calibri\";\n"
"color:black;\n"
"}")
        self.DataPage.setObjectName("DataPage")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.DataPage)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.frame_3 = QtWidgets.QFrame(self.DataPage)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_7 = QtWidgets.QFrame(self.frame_3)
        self.frame_7.setMaximumSize(QtCore.QSize(16777215, 180))
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.frame_9 = QtWidgets.QFrame(self.frame_7)
        self.frame_9.setMinimumSize(QtCore.QSize(0, 0))
        self.frame_9.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_9.setStyleSheet("")
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_9)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton = QtWidgets.QPushButton(self.frame_9)
        self.pushButton.setMinimumSize(QtCore.QSize(100, 0))
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setStyleSheet("QPushButton {\n"
"    color: white;                      /* Text color */\n"
"    padding: 6px 12px;                 /* Padding inside button */\n"
"    font-weight: bold;                 /* Bold text */\n"
"    background-color:#068fff;\n"
"    font:  12pt \"Calibri\";\n"
"    border-radius: 1px;                /* Rounded corners */\n"
"\n"
"\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.frame_9)
        self.pushButton_2.setMinimumSize(QtCore.QSize(100, 0))
        self.pushButton_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_2.setStyleSheet("\n"
"QPushButton {\n"
"    color: white;                      /* Text color */\n"
"    border-radius: 1px;                /* Rounded corners */\n"
"    padding: 6px 12px;                 /* Padding inside button */\n"
"    font-weight: bold;                 /* Bold text */\n"
"background-color:#f2f2f2;\n"
"    font:  12pt \"Calibri\";\n"
"color:#515151;\n"
"\n"
"\n"
"}")
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.verticalLayout_12.addWidget(self.frame_9, 0, QtCore.Qt.AlignHCenter)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_12.addItem(spacerItem)
        self.frame_10 = QtWidgets.QFrame(self.frame_7)
        self.frame_10.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.frame_10)
        self.verticalLayout_13.setContentsMargins(50, 0, 0, 0)
        self.verticalLayout_13.setSpacing(0)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.label_2 = QtWidgets.QLabel(self.frame_10)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(True)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("")
        self.label_2.setObjectName("label_2")
        self.verticalLayout_13.addWidget(self.label_2)
        self.verticalLayout_12.addWidget(self.frame_10)
        self.verticalLayout_6.addWidget(self.frame_7)
        self.frame_8 = QtWidgets.QFrame(self.frame_3)
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.frame_8)
        self.verticalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.frame_11 = QtWidgets.QFrame(self.frame_8)
        self.frame_11.setMaximumSize(QtCore.QSize(16777215, 60))
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_11)
        self.horizontalLayout_4.setContentsMargins(15, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_6 = QtWidgets.QLabel(self.frame_11)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_4.addWidget(self.label_6)
        self.btnBrowse = QtWidgets.QPushButton(self.frame_11)
        self.btnBrowse.setMinimumSize(QtCore.QSize(100, 35))
        self.btnBrowse.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnBrowse.setStyleSheet("QPushButton {\n"
"    color: white;                      /* Text color */\n"
"    padding: 6px 12px;                 /* Padding inside button */\n"
"    font-weight: bold;                 /* Bold text */\n"
"    background-color:#068fff;\n"
"    font:  12pt \"Calibri\";\n"
"    border-radius: 6px;                /* Rounded corners */\n"
"\n"
"\n"
"}")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/images/images/icons/cil-folder-open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnBrowse.setIcon(icon4)
        self.btnBrowse.setIconSize(QtCore.QSize(20, 20))
        self.btnBrowse.setShortcut("")
        self.btnBrowse.setObjectName("btnBrowse")
        self.horizontalLayout_4.addWidget(self.btnBrowse)
        self.verticalLayout_14.addWidget(self.frame_11, 0, QtCore.Qt.AlignLeft)
        self.frame_12 = QtWidgets.QFrame(self.frame_8)
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.frame_12)
        self.verticalLayout_15.setContentsMargins(15, 11, 11, 30)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.dataTable = QtWidgets.QTableWidget(self.frame_12)
        self.dataTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.dataTable.setAlternatingRowColors(True)
        self.dataTable.setShowGrid(True)
        self.dataTable.setRowCount(10)
        self.dataTable.setObjectName("dataTable")
        self.dataTable.setColumnCount(4)
        item = QtWidgets.QTableWidgetItem()
        self.dataTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.dataTable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.dataTable.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.dataTable.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.dataTable.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.dataTable.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.dataTable.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.dataTable.setItem(0, 3, item)
        self.dataTable.horizontalHeader().setCascadingSectionResizes(False)
        self.dataTable.horizontalHeader().setSortIndicatorShown(False)
        self.dataTable.horizontalHeader().setStretchLastSection(True)
        self.dataTable.verticalHeader().setVisible(True)
        self.dataTable.verticalHeader().setCascadingSectionResizes(False)
        self.verticalLayout_15.addWidget(self.dataTable)
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_12)
        self.pushButton_4.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_4.setStyleSheet("QPushButton {\n"
"    color: white;                      /* Text color */\n"
"    border-radius: 6px;                /* Rounded corners */\n"
"    padding: 6px 12px;                 /* Padding inside button */\n"
"    font-weight: bold;                 /* Bold text */\n"
"    background-color: #f2f2f2;\n"
"    font:  12pt \"Calibri\";\n"
"    color: #515151;\n"
"    box-shadow: 0 4px 6px rgba(128, 128, 128, 0.2); /* Grey shadow */\n"
"}\n"
"")
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout_15.addWidget(self.pushButton_4, 0, QtCore.Qt.AlignRight)
        self.verticalLayout_14.addWidget(self.frame_12)
        self.verticalLayout_6.addWidget(self.frame_8)
        self.verticalLayout_7.addWidget(self.frame_3)
        self.stackedWidget.addWidget(self.DataPage)
        self.modelPage = QtWidgets.QWidget()
        self.modelPage.setObjectName("modelPage")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.modelPage)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.frame_4 = QtWidgets.QFrame(self.modelPage)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_4 = QtWidgets.QLabel(self.frame_4)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(20)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("color:black;\n"
"font: 75 20pt \"Calibri\";")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_8.addWidget(self.label_4)
        self.verticalLayout_9.addWidget(self.frame_4)
        self.stackedWidget.addWidget(self.modelPage)
        self.helpPage = QtWidgets.QWidget()
        self.helpPage.setObjectName("helpPage")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.helpPage)
        self.verticalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_11.setSpacing(0)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.frame_6 = QtWidgets.QFrame(self.helpPage)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_5 = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(20)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("color:black;\n"
"font: 75 20pt \"Calibri\";")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_10.addWidget(self.label_5)
        self.verticalLayout_11.addWidget(self.frame_6)
        self.stackedWidget.addWidget(self.helpPage)
        self.verticalLayout_5.addWidget(self.stackedWidget)
        self.horizontalLayout_5.addWidget(self.frame_5)
        self.horizontalLayout_2.addWidget(self.MainScreen)
        MainWindow.setCentralWidget(self.styleSheet)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Active Learning Platform"))
        self.btnHome.setText(_translate("MainWindow", "  Home"))
        self.btnData.setText(_translate("MainWindow", "  Data"))
        self.btnModel.setText(_translate("MainWindow", "  Model"))
        self.btnHelp.setText(_translate("MainWindow", "  Help"))
        self.label_3.setText(_translate("MainWindow", "Home Page"))
        self.pushButton.setText(_translate("MainWindow", "Data"))
        self.pushButton_2.setText(_translate("MainWindow", "Plots"))
        self.label_2.setText(_translate("MainWindow", "Upload the data – you can upload a csv or an excel file.\n"
""))
        self.label_6.setText(_translate("MainWindow", "Select Data File"))
        self.btnBrowse.setText(_translate("MainWindow", "Browse"))
        self.dataTable.setSortingEnabled(False)
        item = self.dataTable.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Column1"))
        item = self.dataTable.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Column2"))
        item = self.dataTable.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Column3"))
        item = self.dataTable.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Column4"))
        __sortingEnabled = self.dataTable.isSortingEnabled()
        self.dataTable.setSortingEnabled(False)
        item = self.dataTable.item(0, 0)
        item.setText(_translate("MainWindow", "Upload"))
        item = self.dataTable.item(0, 1)
        item.setText(_translate("MainWindow", "the"))
        item = self.dataTable.item(0, 2)
        item.setText(_translate("MainWindow", "data by"))
        item = self.dataTable.item(0, 3)
        item.setText(_translate("MainWindow", "Browse button"))
        self.dataTable.setSortingEnabled(__sortingEnabled)
        self.pushButton_4.setText(_translate("MainWindow", "Next Page"))
        self.label_4.setText(_translate("MainWindow", "Model Page"))
        self.label_5.setText(_translate("MainWindow", "Help Page"))
import resources_rc
