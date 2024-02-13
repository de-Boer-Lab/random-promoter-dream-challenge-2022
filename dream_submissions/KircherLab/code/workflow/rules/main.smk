# An example collection of Snakemake rules imported in the main Snakefile.


wildcard_constraints:
    training_type="((training)|(all_training))",

rule splitData:
    conda:
        "../envs/default.yml"
    input:
        script=getInputDataScript("input_data.py"),
        train= getRawTrainDatafile(),
    output:
        train="results/input_data/bin_train.txt.gz",
        test="results/input_data/bin_test.txt.gz",
        validate="results/input_data/bin_validation.txt.gz",
    params:
        seed=getSeed(),
    log:
        "results/logs/splitData.log",
    shell:
        """
        python {input.script} \
        --input {input.train} \
        {params.seed} \
        --output-train {output.train} \
        --output-test {output.test} \
        --output-val {output.validate} &> {log}
        """

rule getAllTraining:
    input:
        "results/input_data/bin_train.txt.gz",
        "results/input_data/bin_test.txt.gz",
        "results/input_data/bin_validation.txt.gz",
    output:
        "results/input_data/bin_all.txt.gz",
    shell:
        """
        zcat {input} | gzip -c > {output}
        """


rule sample:
    conda:
        "../envs/default.yml"
    input:
        script=getInputDataScript("bucket_callibration.py"),
        train=lambda wc: "results/input_data/bin_all.txt.gz"
        if wc.training_type == "all_training"
        else "results/input_data/bin_train.txt.gz",
    output:
        "results/{training_type}_data/{training}.tsv.gz",
    params:
        seed=getSeed(),
        gc_correction=lambda wc: "--gc-correction"
        if useGCCorrection(wc.training)
        else "--no-gc-correction",
        # by config --gc-correction"/--no-gc-correction"
        round_digits=1,
        replacement=lambda wc: "--with-replacement"
        if useReplacement(wc.training)
        else "--without-replacement",
        # by config --with-replacement/--without-replacement
        bucket_frac=lambda wc: "--bucket-frac %f" % getBucketFraction(wc.training)
        if getBucketFraction(wc.training)
        else "",
        bucket_size=lambda wc: "--max-bucket-size %d" % getBucketSize(wc.training)
        if getBucketSize(wc.training)
        else "",
    log:
        "results/logs/sample.{training_type}.{training}.log",
    shell:
        """
        python {input.script} \
        {params.seed} \
        --input {input.train} --output {output} {params.gc_correction} \
        --round-digits {params.round_digits} {params.replacement} \
        {params.bucket_size} {params.bucket_frac} &> {log}
        """


rule train:
    conda:
        "../envs/tensorflow.yml"
    input:
        script=getTrainingScript("train.py"),
        train="results/{training_type}_data/{training}.tsv.gz",
        val="results/input_data/bin_validation.txt.gz",
    output:
        model="results/{training_type}/{training}.{model}.json",  # model structure
        weights="results/{training_type}/{training}.{model}.h5",  # model weights
        val_accuracy="results/{training_type}/{training}.{model}_val_acc.tsv.gz",  # final accuracy of validation output
        fit_log="results/{training_type}/{training}.{model}_fit_log.tsv",  # metric log of each epoch for training and prediction data
    params:
        seed=getSeed(),
        model_type=lambda wc: getModelType(wc.model),  # set up by config. standard/simplified
        model_mode=lambda wc: getModelMode(wc.model), 
        label_columns = lambda wc: getLabelColumns(wc.model),
        batch_size=1024,
        epochs=100,
        learning_rate=0.001,
        loss=lambda wc: getModelLoss(wc.model),  # set up by config. MSE, Huber, Poission, CategoricalCrossentropy [default: MSE]
        tensorboard=lambda wc: getTensorBoard(wc),
        SavedModel=lambda wc: getSavedModel(wc),
        fit_sequence=lambda wc: getFitSequence(wc.training),

    log:
        "results/logs/train.{training_type}.{training}.{model}.log",
    shell:
        """
        python {input.script} --train {input.train} --val {input.val} \
        {params.seed} \
        {params.tensorboard} \
        {params.SavedModel} \
        {params.model_mode} \
        {params.label_columns} \
        {params.fit_sequence} \
        --loss {params.loss} --batch-size {params.batch_size} \
        --model {output.model} --weights {output.weights} \
        --val-acc {output.val_accuracy} --fit-log {output.fit_log}  \
        --epochs {params.epochs} --learning-rate {params.learning_rate} --no-learning-rate-sheduler --use-early-stopping \
        --model-type {params.model_type} &> {log}
        """


rule predict:
    conda:
        "../envs/tensorflow.yml"
    input:
        script=getTrainingScript("predict.py"),
        data=lambda wc: "resources/data_synapse/{data_type}.txt"
        if wc.data_type == "test_sequences"
        else "results/input_data/bin_{data_type}.txt.gz",
        model="results/{training_type}/{training}.{model}.json",  # model structure
        weights="results/{training_type}/{training}.{model}.h5",  # model weights
    output:
        "results/prediction/{training_type}/{training}.{model}.{data_type}.tsv.gz",  # final prediction
    params:
        SavedModel=lambda wc: getSavedModel(wc),
        fit_sequence=lambda wc: getFitSequence(wc.training),
    log:
        "results/logs/predict.{training_type}.{training}.{model}.{data_type}.log",
    shell:
        """
        python {input.script} \
        --test {input.data} \
        {params.SavedModel} \
        --model {input.model} --weights {input.weights} \
        --no-adapter-trimming \
        {params.fit_sequence} \
        --output {output} &> {log}

        """


rule performance:
    conda:
        "../envs/default.yml"
    input:
        script=getTrainingScript("correlation.py"),
        original="results/input_data/bin_{data_type}.txt.gz",
        predicted="results/prediction/training/{training}.{model}.{data_type}.tsv.gz",  # final prediction
    output:
        "results/performance/cor.{training}.{model}.{data_type}.tsv.gz",  # final prediction
    params:
        method="weightedmean",  # TODO set up by config (but weighted mean seems to be the best! so maybe keep it first). maxvalue/weightedmean
        model_mode=lambda wc: getModelMode(wc.model), 
    log:
        "results/logs/performance.{training}.{model}.{data_type}.log",
    shell:
        """
        python {input.script} \
        {params.model_mode} \
        --predicted {input.predicted} \
        --original {input.original} \
        --method {params.method} \
        --output {output} &> {log}
        """


rule leaderboard_submission:
    conda:
        "../envs/default.yml"
    input:
        script=getTrainingScript("create_submission.py"),
        predicted="results/prediction/training/{training}.{model}.test_sequences.tsv.gz",
        original=getRawTestDatafile(),
        sample_submission="resources/sample_submission.json",
    output:
        final="results/final_prediction/{training}.{model}.tsv.gz",  # final prediction of all test_sequences
        leaderboard="results/final_prediction/{training}.{model}.json",  # leaderboard submission
    params:
        method="weightedmean",  # TODO set up by config (but weighted mean seems to be the best! so maybe keep it first). maxvalue/weightedmean
        model_mode=lambda wc: getModelMode(wc.model), 
    log:
        "results/logs/leaderboard_submission.{training}.{model}.log",  # TODO replace wildcards
    shell:
        """
        python {input.script} \
        {params.model_mode} \
        --predicted {input.predicted} \
        --original {input.original} \
        --method {params.method} \
        --output {output.final} \
        --sample-submission {input.sample_submission} {output.leaderboard} &> {log}
        """
