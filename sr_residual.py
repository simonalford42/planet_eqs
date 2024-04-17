num_iters = 5
top_k = 3

def choose_topk(version, top_k):
    '''
    might do angles, might do first and last, slope, etc.
    slope of previous to current is probably good.
    '''
    pass
    # plot the pareto curve for results
    # x = results['complexity']
    # y = results['loss']
    # # plot the pareto frontier
    # plt.scatter(x, y)
    # plt.xlabel('complexity')
    # plt.ylabel('loss')
    # # plt.ymin(1.0)
    # # plt.ymax(6.0)
    # plt.title('pareto frontier')
    # plt.show()

def import_Xy_direct(version):
    # similar to import_Xy_f2, but instead of f2 output, just do the targets (ground truth mean and std)
    pass

def import_Xy_residual(version, residual_model):
    X, y = import_Xy_direct(version)
    # X: the summary stats (outputs of f1, inputs to f2)
    # y: the ground truth targets (mean and std)

    # replace y with the residual
    # something like this in spirit
    # y = y - residual_model.predict(X)

    # these are the "residual targets" for the new round of SR to predict.




for i in range(num_iters):
    # 1. run pysr on the data
    # 21101 is the version of the model with the desired f1 network
    # python sr.py --version 21101 --target direct (this first round would use import_Xy_direct)
    # that spits out a model 87543.pkl, results, etc. with the pareto front

    # 2. choose the top_k models to try from the pareto front
    complexities = choose_topk(version=87543, top_k=top_k)
    # complexities might be [4, 8, 12]: gives the complexity of the models that we're choosing

    # 3. for each model, use it to calculate the residual
    for complexity in complexities:
        # replace with what's actually happening
        # run_pysr(version=...)
        # python sr.py --version 21101 --target residual --residual_model 87543.pkl --complexity 4 (this round would use import_Xy_residual)
        pass

    # then we do the residual of the residual, etc.
