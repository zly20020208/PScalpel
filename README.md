# PScalpel
## Unuploaded data sets required for the project
Some code requires data sets that are too large for the repository to upload. To download unuploaded datasets please click on the following link:

Link: https://pan.baidu.com/s/1QGGHyxLOASi2ABy-GjAWXA

Extraction code: pscp


### 1. System Requirements
- Software Dependencies and Operating Systems：
  - MATLAB R2021a or higher.
  - Operating Systems: Windows 10, macOS 11, Ubuntu 20.04.
    
- Versions the software has been tested on：

  Tested on Windows 10, macOS 11, and Ubuntu 20.04

- Any required non-standard hardware:No special hardware requirements.

### 2. Installation Guide
- Instructions:
  1. Clone the repository:
     ```bash
     git clone https://github.com/zly20020208/PScalpel.git
     ```
  2. Open MATLAB and set the current working directory to the cloned project directory:
     ```matlab
     cd 'path_to_cloned_repository'
     ```

- Typical install time:
  - Approximately 1-2 minutes to set up MATLAB environment and dependencies on a typical desktop computer.

### 3. Demo

This demo shows how to:

1. Compute the phase separation score of any protein sequence.

2. Find a mutated variant with improved phase separation.

3. Find a mutated variant with similar phase separation.

   
- <b>Compute phase separation score:</b><br>
  The compute_score.m script takes a single protein sequence as input and returns its phase separation score.
  - Input：one-letter amino acid sequence
  - Output：a score about phase separation ability
 
    
- <b>Find variant with improved phase separation:</b><br>
  The find_better_variant.m script mutates the input sequence at single positions, evaluates each mutant’s score, and appends the best (highest-scoring) variant to varOfCgasBetter.txt.
  After running, varOfCgasBetter.txt will contain:
    ```php-template
        <best_score>
        <best_variant_sequence>
     ```


 - <b>Find variant with similar phase separation:</b><br>
  The find_same_variant.m script searches for a mutant whose score is closest to the original sequence’s score and appends it to varOfCgasSame.txt.
  After running, varOfCgasSame.txt will contain:
    ```php-template
        <matched_score>
        <matched_variant_sequence>
     ```

     <b>Note:</b>Both find_better_variant.m and find_same_variant.m use the following code to save results (append mode):
    ```matlab
          fp = fopen('varOfCgasSame.txt', 'a');
          fprintf(fp, '%d \n', best_score);
          fprintf(fp, '%s \n', best_seq);
          fclose(fp);
    ```

- <b>Expected run time for demo:</b>
  - Approximately 1-3 minutes on a typical desktop computer, depending on the size of the dataset.


  
