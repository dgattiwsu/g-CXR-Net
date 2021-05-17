# g-CXR-Net: A graphic interface for CXR-Net

A graphic interface to run CXR-Net.

g-CXR-Net is a graphic interface to run CXR-Net two-module Artificial Intelligence pipeline for the quick detection of SARS-CoV-2 from Antero/Posterior (A/P) chest X-rays (CXRs). 

The repository contains two folders, Module_1 and Module_2, required to build a functional CXR-Net model. 

A Python script, Predict_new_patient.py, can be used to run CXR-net in a terminal without gui, selecting from the command line both source/target directories, and, optionally, which dcm image files must be processed for covid/non-covid classification. 

A GUI application, CXR_Net_gui.py is also provided, with python scripts Get_masks.py (running Module I) and Get_scores.py (running Module II) as backends. The app can be launched from a virtual env with the command line syntax: python CXR_Net_gui.py --xdir 'executable_dir', where 'executable_dir' is the directory where CXR_Net_gui.py itself is located. This syntax is retained in order to facilitate the creation of an icon located anywhere in the os tree, linked to the application. The app can also be run outside a virtual environment, but in this case a full path to python location in the virtuel env must be provided. 

A video describing the use of g-CXR-Net can be watched/downloaded from the website listed below.

For additional information e-mail to:

Domenico Gatti

Dept. Biochemistry, Microbiology, and Immunology, Wayne State University, Detroit, MI.

E-mail: dgatti@med.wayne.edu

website: http://veloce.med.wayne.edu/~gatti/
