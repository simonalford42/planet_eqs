def library_learn_example():
    from stitch_core import compress
    expressions = ["(+ 1 (* x1 x2))", "(+ 3 (* x4 x5))", "(+ 5 (* x1 x3))"]
    res = compress(expressions, iterations=1, max_arity=2)
    print(f'result: {res.abstractions}')


library_learn_example()
