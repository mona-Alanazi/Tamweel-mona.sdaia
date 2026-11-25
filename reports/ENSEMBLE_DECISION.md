# قرار التجميع

KEEP SINGLE — Logistic

اخترت النموذج الفردي Logistic Regression (القرار KEEP SINGLE). قارنت ستة مرشحين على نفس 2,155 OOF row عبر 3 forward folds. حقق Logistic أعلى mean AP = 0.392 (fold SD = 0.030)، وكان أفضل من XGBoost (0.353) وLightGBM (0.345)، وحتى من التجميعات: Weighted = 0.389 وStack = 0.383 وEqual = 0.372. شرط الـ Worth-It Gate هو أن يتفوق التجميع على أفضل نموذج فردي بفرق أكبر من الـ fold SD (0.030)، بدون زيادة Brier فوق 0.005 أو ECE فوق 0.01. لم ينجح أي تجميع، لأن الـ lift كان سالبًا (Weighted ‎−0.002، Stack ‎−0.009، Equal ‎−0.020). وفي المعايرة، Logistic حقق Brier = 0.0633 وECE = 0.0188، وWeighted حقق Brier = 0.0633 وECE = 0.0177، يعني تقريبًا نفس النتيجة لكن بتعقيد أكبر (3 نماذج). لذلك التعقيد لا يستحق.

المقارنة استخدمت nested forward OOF: 3 فترات زمنية متتابعة، مع فصل العملاء ونضج الـ labels (90 يومًا)، والـ stacking تعلّم داخل الـ folds فقط. القيود: الـ folds الثلاثة مترابطة زمنيًا وليست مستقلة، والـ SD وصفي وليس confidence interval أو significance test. الـ OOF يغطي 2,155 طلبًا فقط من 6,576 في fit + selection، والصفوف المبكرة warm-up بلا OOF. كما أن اختيار النموذج والعتبة تم على نفس الـ OOF، فالنتائج متفائلة، وهذه البيانات ظهرت في أيام سابقة، فهي ليست untouched holdout.

الدليل: artifacts/ensemble_comparison.csv وday5_ensemble_gate.json. SD وصفي، وليس اختبار دلالة.
