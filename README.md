# RxRx3-ResNet18-Hackathon
Fine-tuning of ResNet18 for binary classification on the RxRx3-core dataset.  Developed for the X Edition Engineering Medicine Hackathon (organised by CEEIBIS and IISGM - Instituto de investigación Sanitaria Gregorio Marañón)

## Summary
Classification between control cells and cells treated with Vinicristine, an oncologic drug used in chemotherapy. It is a first approach, but the long-term aim is to predict terapeutic response to a variety of oncological drugs in order to achieve a more personalized medicine and to reduce time and animal use in experimentation.

## Dataset
RxRx3-core: https://www.rxrx.ai/rxrx3-core

## Pipeline
1. Data extraction from RxRx3-core
2. Preprocess
3. ResNet18 fine-tuning
4. Binary classification
5. Metrics
6. Heatmap (Grad-CAM) for AI explainability

## How to use it
You just have to execute main.py. You can adjust the hyperparameters of the model on the same file.
