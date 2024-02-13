# DreamPromoterExpression

Codes are organized into two jupyter notebooks with instructions on how to run them, these two notebooks are mostly the same except for the data used to train them, see the report for details. You can running through the two notebooks respectively (REAM_NLP_Regression_Train.ipynb and then DREAM_NLP_Regression_Finetune.ipynb) to get the final output.

I am using pytorch and hugginface and ran the code on 2 A40 GPU. The process is fairly straightforward, the trainning step that uses all the training data gets an R2 score around 51-53. And the fine tuning step gets an R2 score around 68-70. This corresponds to 0.628 scored R2 based on the leaderboard feedback.

The repo should contain everything you need to run the models end to end, the saved weights and configs for for my runs were saved on drive at this link:

https://drive.google.com/drive/folders/16JASUUUmoVhMOxqxR4KHHthXKxtFGf0n?usp=sharing

Let me know if you have any questions or issues with the code or process, my email is zlu21@vols.utk.edu.

Thanks for hosting the challenge, it's been fun!

