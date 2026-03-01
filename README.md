# RxRx3-ResNet18-Hackathon
Fine-tuning of ResNet18 for binary classification on the RxRx3-core dataset.  Developed for the X Edition Engineering Medicine Hackathon (organised by CEEIBIS and IISGM - Instituto de investigación Sanitaria Gregorio Marañón)

## Summary
Classification between control cells and cells treated with Vinicristine, an oncologic drug used in chemotherapy. While this is a first approach, the long-term aim is to predict terapeutic response to a variety of oncological drugs in order to achieve a more personalized medicine and to reduce time and animal use in experimentation.

## Dataset
RxRx3-core: https://www.rxrx.ai/rxrx3-core
The dataset contains HCS images for representation learning and biological perturbation analysis

## How to use it
1. Install dependencies: pip install -r requirements.txt
2. Execute main.py. Hyperparameters can be adjusted directly within the scipt.

## Pipeline
1. Data extraction from RxRx3-core
2. Preprocess
3. ResNet18 fine-tuning
4. Binary classification
5. Metrics
6. Heatmap (Grad-CAM) for AI explainability. Highlights the regions the model focuses on to make its prediction.
<img width="736" height="490" alt="pipeline" src="https://github.com/user-attachments/assets/9f4070cf-ec0d-48b1-b03e-901a897f38a4" />

## Technical details
1. Class balancing. To avoid bias, 192 Vinicristine images were exactly matched with 192 empty-control images. 
2. Cropping. Original 512x512 images were divided into four 224x224 images. This was part of the normalization required by ResNet18, plus it addressed image scarcity, resulting in 1536 usable image samples.
3. Data leakage prevention. Splitting (Train/Val/Test) was performed before cropping. This assures that crops from the same original image never appear in different splits.
4. Split. Standard 70%/15%/15% distribution
5. Architecture. Only the final fully connected layer of the ResNet18 was modified to match our binary output.

## Limitations
1. Binary vs. Multiclass. Currently limited to Vincristine detection. Future iterations should include multi-drug classification.
2. Cell Type. The RxRx3-core dataset uses HUVEC (Human Umbilical Vein Endothelial Cells), which are not tumor cells. Testing on actual cancer cell lines is a logical next step.
3. Lack of diversity. Since the fine-tuning was performed on a relatively small subset of 1536 crops, the model may not generalize well to other cell lines or different microscopy imaging settings
4. Static snapshot vs dynamic processes. The images represent a single point in time. However, chemotherapy response is a dynamic process. A static binary classification ignores the temporal evolution of cell death or morphological change.
