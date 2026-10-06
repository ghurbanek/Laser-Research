[WIP]

Hello! This project is in fulfillment of the Undergraduate Research required for physics majors at Arkansas Tech University, performed under the advisorship of Dr. Jessica Young. This description will become more thorough as more things happen, but in short the goal is to leverage machine learning techniques for the calibration/alignment of a spatial light modulator (SLM), and perhaps correct for non-Gaussian-ness in the initial beam. Time will tell!

Note: To ensure correct package dependencies, create a new conda environment with Python 3.13.5 by running ``conda env create -f environment.yml`` from a terminal in a local clone of this repository. This creates a new environment named "BeamShaping" in which all the requisite packages will be automatically installed. If/when the environment gets updated (new package needed), run ``conda env update -f environment.yml --prune``

We used a Thorlabs DCC1645C CMOS Camera from, and as such any code interfacing with the camera will require the ``uc480_64.dll`` driver, available from https://www.thorlabs.com/software_pages/viewsoftwarepage.cfm?code=ThorCam under the "ThorCam Software" tab.

Finally, all datasets in the repository are stored using Git LFS. Running a typical `git pull` or `git clone` will get all the other files, but only *pointers* to the data itself, so one will likely also need to run `git lfs pull` to get the .npz files in which our image data is stored.

