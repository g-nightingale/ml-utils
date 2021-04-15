class MonteCarloCV:
    """
    Monte Carlo Cross Validation.
    """

    def __init__(self, models, pct_train=0.7, number_of_runs=100, random_seed=42):

        self.models = models
        self.pct_train = pct_train
        self.number_of_runs = number_of_runs
        self.random_seed = random_seed
        self.results_dict = {}
        self.train_meta_scores = []
        self.val_meta_scores = []
        self.train_meta_costs = []
        self.val_meta_costs = []

    def train(self, x, y, feature_processing=None, verbose=False, verbose_n=10, metric='auc', ):
        """
        Train the models.
        """
        n_train = int(len(x) * self.pct_train)

        for m, model in enumerate(self.models):
            if verbose:
                print(f'Model {m}')
            train_scoring = []
            val_scoring = []
            train_costs_model = []
            val_costs_model = []

            x_copy = x.copy()
            y_copy = y.copy()

            seed = self.random_seed

            for i in range(self.number_of_runs):

                np.random.seed(seed)
                np.random.shuffle(x_copy)
                np.random.seed(seed)
                np.random.shuffle(y_copy)
                seed += 1

                x_train = x_copy[:n_train]
                x_val = x_copy[n_train:]

                y_train = y_copy[:n_train]
                y_val = y_copy[n_train:]

                if feature_processing is not None:
                    x_train, x_val = feature_processing(x_train, x_val)

                model.fit(x_train, y_train)

                if metric == 'auc':
                    train_score = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
                    val_score = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
                else:
                    train_score = accuracy_score(y_train, model.predict(x_train))
                    val_score = accuracy_score(y_val, model.predict(x_val))

                train_scoring.append(train_score)
                val_scoring.append(val_score)

                if verbose and i % verbose_n == 0:
                    print(f'Iteration {i} - cumulative average score: {np.mean(val_scoring)}')

            if verbose:
                print(f'Model {m} average accuracy: {np.mean(val_scoring)} \n')

            self.results_dict[m] = (np.mean(val_scoring), np.std(val_scoring))
            self.train_meta_scores.append(train_scoring)
            self.val_meta_scores.append(val_scoring)
            self.train_meta_costs.append(train_costs_model)
            self.val_meta_costs.append(val_costs_model)

    def train_params(self, x, y, feature_processing=None, params=None, verbose=False, verbose_n=10):
        """
        Train the models with user supplied parameters
        """
        n_train = int(len(x) * self.pct_train)
        c = 0

        for m, model in enumerate(self.models):
            if verbose:
                print(f'Model {m}')
            train_scoring = []
            val_scoring = []
            train_costs_model = []
            val_costs_model = []

            x_copy = x.copy()
            y_copy = y.copy()

            seed = self.random_seed

            param = params[m]
            stem = param['stem']
            count_vectorizer = param['count_vectorizer']
            min_df = param['min_df']
            ngram_range = param['ngram_range']

            for i in range(self.number_of_runs):

                np.random.seed(seed)
                np.random.shuffle(x_copy)
                np.random.seed(seed)
                np.random.shuffle(y_copy)
                seed += 1

                x_train = x_copy[:n_train]
                x_val = x_copy[n_train:]

                y_train = y_copy[:n_train]
                y_val = y_copy[n_train:]

                # Feature processing
                x_train, x_val = feature_processing(x_train, x_val, stem=stem, count_vectorizer=count_vectorizer,
                                                    min_df=min_df, ngram_range=ngram_range)

                model.fit(x_train, y_train)

                if metric == 'auc':
                    train_score = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
                    val_score = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
                else:
                    train_score = accuracy_score(y_train, model.predict(x_train))
                    val_score = accuracy_score(y_val, model.predict(x_val))

                train_scoring.append(train_score)
                val_scoring.append(val_score)

                if verbose and i % verbose_n == 0:
                    print(f'Iteration {i} - cumulative average score: {np.mean(val_scoring)}')

            if verbose:
                print(f'Model {m} - average accuracy: {np.mean(val_scoring)} \n')

            self.results_dict[m] = (np.mean(val_scoring), np.std(val_scoring))
            self.train_meta_scores.append(train_scoring)
            self.val_meta_scores.append(val_scoring)
            self.train_meta_costs.append(train_costs_model)
            self.val_meta_costs.append(val_costs_model)

    def plot_scores(self, colors=None, labels=None, bins=20, xlim=(0.0, 1.0)):
        """
        Plot the density functions of the model scores.
        """

        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        plt.title(f'Monte Carlo CV Density Function - Results over {self.number_of_runs} runs',
                  fontsize=14, weight='bold')
        for m, scores in enumerate(self.val_meta_scores):
            if labels is None:
                label = ''
            else:
                label = labels[m] + ' - mean: ' + (str(round(np.mean(scores), 4))) + ' stdev: ' + (
                    str(round(np.std(scores), 4)))
            if colors is None:
                sns.kdeplot(scores, label=label, fill=True, alpha=0.25)
            else:
                sns.kdeplot(scores, color=colors[m], label=label, fill=True, alpha=0.25)
        plt.xlabel('validation accuracy')
        plt.ylabel('frequency')
        plt.xlim(xlim)
        if labels is not None:
            plt.legend(loc='upper left', bbox_to_anchor=(0.0, -0.1), frameon=False)
        plt.show()