Infos about the model are in the Thesis
all the important files and scripts are in this folder. It is important to follow the right execution order, otherwise the scipt will fail to provide a result.
The following order has to be fulfulled for a satisfy results.
- 01_create_mastertable.py
- 02_particle_based_transport_model.py
- 03_model_evaluation_metrics.py

01_create_mastertable.py
This file loads in the cleaned data from the tracer experiments and calculates the metrics that will be used by the model and have been analyzed in the first part of the thesis.
It lays the foundation of the model and evaluation process.

02_particle_based_transport_model.py
This file is the implementation of the model descibed in the thesis. 


03_model_and_evaluation.py
This file runs the model and calculates the metrics for each injection and station along the stream

