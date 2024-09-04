### Research on Music Models

This research explores various models that work with music data. The music is represented in two possible formats:

1. **Time Steps Representation:**
   - Each time step consists of:
     - **Pitch**
     - **Duration** (the length of time the pitch is held)
     - **Offset** (location of note in time)

2. **One-Hot Encoding Representation:**
   - Pitches are represented using one-hot encoding.
   - The difference in time between the steps is a constant value (the smallest time difference in the song).

### Models Used

The research involves a variety of models, ranging from statistical methods to standard machine learning approaches. However, the primary focus is on **Tangled Program Graphs (TPG)**.

### TPG Customization

- The TPG library was forked and modified to return a single action.
- Preliminary experiments are being conducted with curriculum learning.
   - The training, validation, and testing schemes are based on the smoothness of the data.

### Data Division Methodology

The methodology for data division specifically tailored for TPG can be found [here](https://ali-naqvi.ca/uploads/Towards_Creativity.pdf).
