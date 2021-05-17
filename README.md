g-CXR-Net: A graphic interface for CXR-Net

Haikal Abdulah 1,4, Benjamin Huber 1,4, Hassan Abdallah 3, Luigi L. Palese 4, Hamid Soltanian-Zadeh 5, Domenico L. Gatti 1,2,6*

1 Department of Biochemistry, Microbiology and Immunology, Wayne State Univ., Detroit, MI, USA

2 NanoBioScience Institute, Wayne State Univ., Detroit, MI, USA

3 Department of Biostatistics, University of Michigan, Ann Arbor, MI, USA

4 Department of Basic Medical Sciences, Neurosciences and Sense Organs, Univ. of Bari Aldo Moro, Bari, Italy

5 Departments of Radiology and Research Administration, Henry Ford Health System, Detroit, MI, USA

6 Molecular Medicine Institute, Cambridge, MA 02138,USA

*E-mail: dgatti@med.wayne.edu

g-CXR-Net is a graphic interface to run CXR-Net two-module Artificial Intelligence pipeline for the quick detection of SARS-CoV-2 from Antero/Posterior (A/P) chest X-rays (CXRs). 

The repository contains two folders, Module_1 and Module_2, required to build a functional CXR-Net model. 

A GUI application, CXR_Net_gui.py is also provided, with python scripts Get_masks.py (running Module I) and Get_scores.py (running Module II) as backends. The GUI app can be launched from a virtual env with the command line syntax: python CXR_Net_gui.py --xdir 'executable_dir', where 'executable_dir' is the directory where CXR_Net_gui.py itself is located. This syntax is retained in order to facilitate the creation of an icon located anywhere in the os tree, linked to the application.

A video describing the use of g-CXR-Net can be watched/downloaded from the website listed below.

For additional information e-mail to:

Domenico Gatti

Dept. Biochemistry, Microbiology, and Immunology, Wayne State University, Detroit, MI.

E-mail: dgatti@med.wayne.edu

website: http://veloce.med.wayne.edu/~gatti/
