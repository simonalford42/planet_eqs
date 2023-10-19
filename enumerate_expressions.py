import itertools
import random
import numpy as np
import math
import utils

class Expression():

    def evaluate(self, environment):
        return self.__class__.op(*[arg.evaluate(environment) for arg in self.arguments()])

    def arguments(self):
        assert False, "not implemented"

    def cost(self):
        return 1 + sum([0] + [argument.cost() for argument in self.arguments()])

    def pretty_print(self):
        return '(' + self.__class__.op_str + ' ' + ' '.join([arg.pretty_print() for arg in self.arguments()]) + ')'

    def __str__(self):
        return self.__class__.__name__ + '(' + ', '.join(str(a) for a in self.arguments()) + ')'


    @property
    def return_type(self):
        return self.__class__.return_type

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self): return hash(str(self))

    def __ne__(self, other): return str(self) != str(other)

    def __gt__(self, other): return str(self) > str(other)

    def __lt__(self, other): return str(self) < str(other)


class Real(Expression):
    return_type = "real"
    argument_types = []

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Real('{self.name}')"

    def pretty_print(self):
        return self.name

    def evaluate(self, environment):
        return environment[self.name]

    def arguments(self): return []

class Vector(Expression):
    return_type = "vector"
    argument_types = []

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Vector('{self.name}')"

    def pretty_print(self):
        return self.name

    def evaluate(self, environment):
        return environment[self.name]

    def arguments(self): return []


class Plus(Expression):
    return_type = "real"
    argument_types = ["real","real"]
    op_str = "+"
    op = lambda x, y: x + y

    def __init__(self, x, y):
        self.x, self.y = x, y

    def arguments(self): return [self.x, self.y]

class Times(Expression):
    return_type = "real"
    argument_types = ["real","real"]
    op_str = "*"
    op = lambda x, y: x * y

    def __init__(self, x, y):
        self.x, self.y = x, y

    def arguments(self): return [self.x, self.y]

class Divide(Expression):
    return_type = "real"
    argument_types = ["real","real"]
    op_str = "/"
    op = lambda x, y: x / y

    def __init__(self, x, y):
        self.x, self.y = x, y

    def arguments(self): return [self.x, self.y]


class Inner(Expression):
    return_type = "real"
    argument_types = ["vector","vector"]
    op_str = "dp"

    def __init__(self, x, y):
        self.x, self.y = x, y

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        if isinstance(x, np.ndarray):
            return np.sum(x * y, -1)
        return (x * y).sum(-1).unsqueeze(-1)

    def arguments(self): return [self.x, self.y]


class AbstractionConst(Expression):
    return_type = "vector"
    argument_types = []

    def __init__(self):
        self.expr = ScaleInverse(Hat(Vector('R')), Inner(Vector('R'), Vector('R')))

    def __str__(self):
        return f"Abstraction2()"

    def pretty_print(self):
        return f"##"

    def evaluate(self, environment):
        return self.expr.evaluate(environment)

    def arguments(self): return []


class Cross(Expression):
    return_type = "vector"
    argument_types = ["vector","vector"]
    op_str = "X"

    def __init__(self, x, y):
        self.x, self.y = x, y

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, np.ndarray)
        return np.cross(x, y)

    def arguments(self): return [self.x, self.y]

class Outer(Expression):
    return_type = "matrix"
    argument_types = ["vector","vector"]
    op_str = "op"

    def __init__(self, x, y):
        self.x, self.y = x, y

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, np.ndarray)
        return np.outer(x, y)

    def arguments(self): return [self.x, self.y]

class Skew(Expression):
    return_type = "matrix"
    argument_types = ["vector"]
    op_str = "skew"

    def __init__(self, x):
        self.x = x

    def evaluate(self, environment):
        v = self.x.evaluate(environment)
        assert isinstance(v, np.ndarray)
        x, y, z = v[0], v[1], v[2]

        return np.array([[0, -z, y],
                         [z, 0, -x],
                         [-y, x, 0]])

    def arguments(self): return [self.x]


class Sin(Expression):
    return_type = "real"
    argument_types = ["real"]
    op_str = "sin"
    op = lambda x: np.sin(x)

    def __init__(self, x):
        self.x = x

    def arguments(self): return [self.x]


class Cos(Expression):
    return_type = "real"
    argument_types = ["real"]
    op_str = "cos"
    op = lambda x: np.cos(x)

    def __init__(self, x):
        self.x = x

    def arguments(self): return [self.x]


class SinCosVec(Expression):
    return_type = "vector"
    argument_types = ["vector"]
    op_str = "scv"

    def op(v):
        assert v.shape[0] == 2
        return np.array([np.sin(v[0]), np.cos(v[1])])

    def __init__(self, x):
        self.x = x

    def arguments(self): return [self.x]


class Length(Expression):
    return_type = "real"
    argument_types = ["vector"]
    op_str = "len"
    op = lambda x: np.sum(x * x)**0.5

    def __init__(self, x):
        self.x = x

    def arguments(self): return [self.x]


class Perp(Expression):
    return_type = "vector"
    argument_types = ["vector"]
    op_str = "perp"
    op = lambda x: np.array([-x[1], x[0]])

    def __init__(self, x):
        self.x = x

    def arguments(self): return [self.x]


class Within100(Expression):
    return_type = "real"
    argument_types = ["real"]
    op_str = "within100"
    op = lambda x: 1 if x < 100 else 0

    def __init__(self, x):
        self.x = x

    def arguments(self): return [self.x]


class Within35(Expression):
    return_type = "real"
    argument_types = ["real"]
    op_str = "within35"
    op = lambda x: 1 if x < 35 else 0

    def __init__(self, x):
        self.x = x

    def arguments(self): return [self.x]



class Scale(Expression):
    return_type = "vector"
    argument_types = ["real","vector"]
    op_str = "v*"
    op = lambda x, y: x * y

    def __init__(self, x, y):
        self.x, self.y = x, y

    def arguments(self): return [self.x, self.y]


class ScaleInverse(Expression):
    return_type = "vector"
    argument_types = ["vector", "real"]
    op_str = "v/"
    op = lambda x, y: x / y

    def __init__(self, x, y):
        self.x, self.y = x, y

    def arguments(self): return [self.x, self.y]

class Hat(Expression):
    return_type = "vector"
    argument_types = ["vector"]
    op_str = "hat"

    def __init__(self, x):
        self.x = x

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        if isinstance(x, np.ndarray):
            norm = np.sum(x * x, -1)**0.5
            if norm > 0: return x/norm
            else: return np.zeros(x.shape)
        else:
            return x/(((x*x).sum(-1)**0.5).unsqueeze(-1))

    def arguments(self): return [self.x]

class RInRadius4(Expression):
    return_type = "real"
    argument_types = []
    op_str = "rinradius4"

    def __init__(self):
        self.r = Vector('R')

    def evaluate(self, environment):
        r = self.r.evaluate(environment)
        return 1 if np.sum(r * r)**0.5 < 4 else 1E-5
        # return 1 / (1 + np.exp(10*(r-5)))

    def arguments(self): return []

class RInRadius48(Expression):
    return_type = "real"
    argument_types = []
    op_str = "rinradius48"

    def __init__(self):
        self.r = Vector('R')

    def evaluate(self, environment):
        r = self.r.evaluate(environment)
        return 1 if np.sum(r * r)**0.5 < 48 else 1E-5
        # return 1 / (1 + np.exp(10*(r-5)))

    def arguments(self): return []


def abstraction(expr):
    class Abstraction(Expression):
        return_type = expr.__class__.return_type
        argument_types = []

        def __init__(self):
            self.expr = expr

        def __str__(self):
            return f"Abstraction({expr})"

        def pretty_print(self):
            return f"##"

        def arguments(self): return []

        def evaluate(self, environment):
            return self.expr.evaluate(environment)

    return Abstraction()


def expr_structure(expr):
    '''
    Helps disambiguate some of the pretty print structures
    '''
    if type(expr) == Vector or type(expr) == Real:
        return type(expr).__name__ + '(' + str(expr.name) + ')'

    # print class name, parenthesis, and then recurse on the argumentsj
    return (type(expr).__name__
            + '('
            + ', '.join([expr_structure(arg) for arg in expr.arguments()])
            + ')')


def all_operators():
    return [Plus, Times, Divide, Inner, Cross, Outer, Skew, Length, Scale, ScaleInverse, Hat]

def pretty_print_to_expr(expr_str):
    # example: "(+ (len V) (len V))" becomes Plus(Length(Vector('V')), Length(Vector('V')))
    # (+ (len V) (len V)) becomes Plus(Length(Vector('V')), Length(Vector('V')))
    # (op (X R V1) (/ (hat R) (* (dp R R) (dp R R))))
    # (skew (/ V1 (* (dp R R) (dp R (hat R)))))
    op_str_to_op = {op.op_str: op for op in all_operators()}
    # we don't support Reals yet
    const_str_to_const = {s: Vector(s) for s in ['R', 'V', 'V1', 'V2']}
    const_str_to_const["##"] = AbstractionConst()

    def parse_expr(expr_str):
        if expr_str[0] == '(' and expr_str[-1] == ')':
            op_str = expr_str[1:].split(' ')[0]
            args_str = expr_str[len(op_str)+2:-1]
            # args are either surrounded by parentheses, or a word like R or V1
            args = []
            while len(args_str) > 0:
                if args_str[0] == '(':
                    # find the matching closing parenthesis
                    depth = 0
                    for i, c in enumerate(args_str):
                        if c == '(': depth += 1
                        if c == ')': depth -= 1
                        if depth == 0:
                            break
                    arg_str = args_str[:i+1]
                    args.append(parse_expr(arg_str))
                    args_str = args_str[i+2:]
                else:
                    if ' ' not in args_str:
                        arg_str = args_str
                        args.append(parse_expr(arg_str))
                        break
                    else:
                        i = args_str.index(' ')
                        arg_str = args_str[:i]
                        args.append(parse_expr(arg_str))
                        args_str = args_str[i+1:]

            return op_str_to_op[op_str](*args)
        else:
            assert expr_str in const_str_to_const, f'unknown constant {expr_str}'
            return const_str_to_const[expr_str]

    return parse_expr(expr_str)


def weighted_bottom_up_generator(cost_bound, operators, constants, inputs, cost_dict=None):
    """
    cost_bound: int. an upper bound on the cost of an expression
    operators: list of classes, such as [Times, If, ...]
    constants: list of possible leaves in syntax tree, such as [Number(1)]. Variables can also be leaves, but these are automatically inferred from `input_outputs`
    inputs: list of environments (the input), such as [{'x': 5}, {'x': 1}]
    cost_dict: dict mapping Expression subclass to their cost. costs must be integral. if none provided, does uniform cost (1 per node)
    yields: sequence of programs, ordered by expression cost, which are semantically distinct on the input examples
    """

    variables = infer_variables(inputs)
    variables_and_constants = constants + variables

    if cost_dict is None:
        cost_dict = {op: 1 for op in operators}
        cost_dict.update({type(v): 1 for v in variables_and_constants})
    else:
        cost_bound = None

    for op in operators:
        if op not in cost_dict:
            cost_dict[op] = 1
            print(f"warning: no cost provided for {op}. using cost 1")

    for v in variables_and_constants:
        if type(v) not in cost_dict:
            cost_dict[type(v)] = 1
            print(f"warning: no cost provided for {v}. using cost 1")

    assert all(isinstance(cost, int) for cost in cost_dict.values()), 'costs must be integral'

    # a mapping from a tuple of (type, expression_size) to all of the possible values that can be computed of that type using an expression of that size
    observed_values = set()

    enumerated_expressions = {}
    def record_new_expression(expression, cost):
        """Returns True iff the semantics of this expression has never been seen before"""
        nonlocal inputs, observed_values

        try:
            valuation = np.array([expression.evaluate(input) for input in inputs])
        except ZeroDivisionError:
            return False

        if np.max(np.abs(valuation)) < 1e-5:
            return False # bad expression

        if np.any(~np.isfinite(valuation)):
            return False

        # we only care about the direction, not rescaling or sign changes
        valuation = valuation / np.linalg.norm(valuation)

        v1 = np.around(valuation*100, decimals=5).tobytes()
        v2 = np.around(-valuation*100, decimals=5).tobytes()

        if v1 not in observed_values and v2 not in observed_values:
            observed_values.add(v1)

            # we have some new behavior
            key = (expression.__class__.return_type, cost)

            if key not in enumerated_expressions:
                enumerated_expressions[key] = []

            enumerated_expressions[key].append( (expression, (v1, len(observed_values))))

            return True

        return False

    lvl = 0
    while True:
        for terminal in variables_and_constants:
            if cost_dict[type(terminal)] == lvl:
                if record_new_expression(terminal, lvl):
                    yield terminal

        for operator in operators:
            op_cost = cost_dict[operator]
            if op_cost < lvl:
                for arg_costs in integer_partitions(lvl - op_cost, len(operator.argument_types)):
                    candidate_arguments = [enumerated_expressions.get((typ, cost), [])
                                           for typ, cost in zip(operator.argument_types, arg_costs)]
                    for arguments in itertools.product(*candidate_arguments):
                        # note: could make use of precomputed evaluation of
                        # subtrees when evaluating later
                        new_expression = operator(*[e for e,v in arguments])
                        if record_new_expression(new_expression, lvl):
                            # if len(observed_values) % 1000 == 0:
                                # print(f'{len(observed_values)} expressions')
                            yield new_expression

        lvl += 1
        if cost_bound and lvl > cost_bound:
            break


def integer_partitions(target_value, number_of_arguments):
    """ Returns all ways of summing up to `target_value` by adding
        `number_of_arguments` nonnegative integers.
     """

    if target_value < 0:
        return []

    if number_of_arguments == 1:
        return [[target_value]]

    return [ [x1] + x2s
             for x1 in range(target_value + 1)
             for x2s in integer_partitions(target_value - x1, number_of_arguments - 1) ]


def fit_pcfg(expressions, operators, pseudocount=1):
    '''
     Returns a cost dict for the PCFG that is a good fit for the given expressions.
     Infers variables from those present in expressions.

     pseudocount is added to the count of each operator, to avoid zero probabilities.
    '''

    def get_variables(expression):
        if isinstance(expression, Vector) or isinstance(expression, Real):
            return {type(expression)}
        else:
            return set().union(*[get_variables(arg) for arg in expression.arguments()])

    variables = set().union(*[get_variables(expr) for expr in expressions])
    vars_and_ops = list(variables) + operators

    # the "counts" are the number of times that expression occurs
    # some expressions might occur more than others because that type is more common or something.
    # as a start, normalize by the number of times expressions of that type occur
    counts = {op: pseudocount for op in vars_and_ops}
    type_counts = {op.return_type: 0 for op in vars_and_ops}

    # add pseudocount to the denominator for each op of that type
    for op in vars_and_ops:
        type_counts[op.return_type] += pseudocount

    def calc_counts(expressions):
        for expr in expressions:
            counts[type(expr)] += 1
            calc_counts(expr.arguments())

    def calc_type_counts(expressions):
        for expr in expressions:
            type_counts[expr.return_type] += 1
            calc_type_counts(expr.arguments())

    calc_counts(expressions)
    calc_type_counts(expressions)

    # print(f'{counts=}')
    # print(f'{type_counts=}')

    # probability = count / type_count
    # cost = rounded negative log of probability
    cost_dict = {op: -math.log(count / type_counts[op.return_type]) for op, count in counts.items()}
    # directly rounding is too coarse, so instead round to nearest 1/10th.
    # then multiply by 10 so costs are still integers
    cost_dict = {op: round(cost * 10) for op, cost in cost_dict.items()}

    # don't allow cost zero... add one to each
    cost_dict = {op: cost + 1 for op, cost in cost_dict.items()}

    return cost_dict


def infer_variables(inputs):
    def make_variable(variable_name, variable_value):
        if isinstance(variable_value, float): return Real(variable_name)
        if isinstance(variable_value, np.ndarray): return Vector(variable_name)

    variables = list({make_variable(variable_name, variable_value)
                      for input in inputs
                      for variable_name, variable_value in input.items() })
    # since inputs are randomly generated, this ensures a consistent order
    # so that the synthesis order is deterministic
    variables = sorted(variables, key=lambda v: v.name)
    return variables


basis_cache = {}
def construct_basis(reals, vectors, size, operators, dimension=3, cost_dict=None, cost_bound=20, inputs=None):
    basis_key = (tuple(reals), tuple(vectors), size, dimension)
    if cost_dict is None and basis_key in basis_cache: return basis_cache[basis_key]

    # operators = get_operators(reals, vectors, dimension)

    def random_input():
        d = {}
        for nm in reals:
            d[nm] = random.random()*10
        for nm in vectors:
            d[nm] = np.random.random(dimension)*10-5
        return d

    if inputs == None:
        inputs = []
    # if other inputs provided, use random inputs too to ensemble the coverage..
    inputs += [random_input() for _ in range(10)]

    vector_basis = []
    matrix_basis = []

    variables = infer_variables(inputs)
    constants = []

    # add any constants from the op list to the list of constants
    old_operators = operators
    operators = []
    for op in old_operators:
        # library learned constants aren't classes, but actual instances of exprs
        if isinstance(op, Expression):
            constants.append(op)
        elif op.argument_types == []:
            constants.append(op())
        else:
            operators.append(op)

    for expression in weighted_bottom_up_generator(cost_bound, operators, constants,
                                                  inputs, cost_dict=cost_dict):
        if expression.return_type == "vector" and len(vector_basis) < size:
            vector_basis.append(expression)
            # if len(vector_basis) % 100 == 0:
                # print(f'{len(vector_basis)} vector basis terms')

        if expression.return_type == "matrix" and len(matrix_basis) < size:
            matrix_basis.append(expression)
            # if len(matrix_basis) % 100 == 0:
                # print(f'{len(matrix_basis)} matrix basis terms')

        if len(vector_basis) >= size and len(matrix_basis) == 0: break
        if len(vector_basis) >= size and len(matrix_basis) >= size: break

    basis_cache[basis_key] = (vector_basis, matrix_basis)
    return basis_cache[basis_key]


def dipole_cost_dict():
    operators = get_operators()
    cost_dict = {op: 100 for op in operators}

    R = Vector('R')
    V1 = Vector('V1')

    if Norm in operators:
        assert NormFifth not in operators
        assert NormCubed not in operators
        p1 = Skew(ScaleInverse(V1, Times(Norm(R), Length(R))))
        p2 = Outer(Cross(R, V1), ScaleInverse(Hat(R), Times(Norm(R), Norm(R))))
    elif AbstractionConst in operators:
        p1 = Skew(ScaleInverse(V1, Divide(Length(R), Length(AbstractionConst()))))
        p2 = Outer(Cross(AbstractionConst(), V1), Scale(Length(AbstractionConst()), R))
    # when we have norm forth and norm cubed
    elif NormFifth in operators and NormCubed in operators:
        p1 = Skew(ScaleInverse(V1, NormCubed(R)))
        p2 = Outer(Cross(R, V1), ScaleInverse(R, NormFifth(R)))
    # when we only have norm fifth
    elif NormFifth in operators and NormCubed not in operators and NormForth not in operators:
        p1 = Skew(ScaleInverse(V1, Length(Scale(Inner(R, R), R))))
        p2 = Outer(Cross(R, V1), ScaleInverse(R, NormFifth(R)))
    # when we only have norm cubed
    elif NormCubed in operators and NormFifth not in operators and NormForth not in operators:
        p1 = Skew(ScaleInverse(V1, NormCubed(R)))
        p2 = Outer(Cross(V1, Hat(R)), ScaleInverse(Hat(R), NormCubed(R)))
    # when we have no norms
    elif NormCubed not in operators and NormFifth not in operators and NormForth not in operators:
        p1 = ScaleInverse(Outer(Cross(V1, R), Hat(R)), Times(Inner(R, R), Inner(R, R)))
        # p2 = Skew(ScaleInverse(V1, Length(Scale(Inner(R, R), R))))
        p2 = None
    else:
        raise ValueError('unknown operator combination')

    exprs = [p1 for _ in range(10)]
    if p2 is not None:
        exprs += [p2 for _ in range(10)]

    cost_dict = fit_pcfg(exprs, get_operators(), pseudocount=0.40)

    # cost_dict[Vector] = 5
    cost_dict[Outer] = 2
    # cost_dict[ScaleInverse] = 24
    # cost_dict[Hat] = 24
    # cost_dict[Length] = 45
    # cost_dict[Scale] = 57
    # cost_dict[Inner] = 5
    # cost_dict[Times] = 12
    # cost_dict[Skew] = 34
    # cost_dict[Cross] = 24

    return cost_dict


def all_task_solution_expressions_with_abstraction_const():
    # drag3
    # (R_hat / R^2)
    # (* (len V) V)

    # falling
    # (R_hat / R^2)
    # (R_hat / R^2)

    # orbit
    # (R_hat / R^2)
    # (R_hat / R^2)

    # orbit2
    # (R_hat / R^2)
    # (R_hat / R^2)
    # (R_hat / R^2)
    # (R_hat / R^2)
    # (R_hat / R^2)
    # (R_hat / R^2)
    # (* (dp (R_hat / R^2) (R_hat / R^2)) (R_hat / R^2))
    # (* (len (R_hat / R^2)) (R_hat / R^2))

    # drag1
    # V

    # drag2
    # (* (len V) V)

    # magnet1
    # (skew V)

    # magnet2
    # (op (X (R_hat / R^2) V1) (R_hat / R^2))
    # (* (dp (R_hat / R^2) (R_hat / R^2)) (R_hat / R^2))
    # (op (X (R_hat / R^2) V1) (R_hat / R^2))
    # (op (R_hat / R^2) (X (R_hat / R^2) V1))
    # (op (X (R_hat / R^2) V1) (R_hat / R^2))
    # (op (R_hat / R^2) (X (R_hat / R^2) V1))
    # (X (R_hat / R^2) V1)
    # (skew (* (len (R_hat / R^2)) V1))
    # (op (R_hat / R^2) (X (R_hat / R^2) V1))
    # (skew (* (len (R_hat / R^2)) V1))
    # (op (X (R_hat / R^2) V1) R)
    # (op (X (R_hat / R^2) V1) R)
    # (skew (* (len (R_hat / R^2)) V1))
    # (op (X (R_hat / R^2) V1) R)
    # (* (dp R V1) V1)
    # (X R V1)
    # (op (X R V1) V1)
    # (op (X R V1) V1)
    R = Vector('R')
    V = Vector('V')
    V1 = Vector('V1')

    drag3_1 = AbstractionConst()
    drag3_2 = Times(Length(V), V)

    falling1 = AbstractionConst()
    falling2 = AbstractionConst()

    orbit1_1 = AbstractionConst()
    orbit1_2 = AbstractionConst()

    orbit2_1 = AbstractionConst()
    orbit2_2 = AbstractionConst()
    orbit2_3 = AbstractionConst()
    orbit2_4 = AbstractionConst()
    orbit2_5 = AbstractionConst()
    orbit2_6 = AbstractionConst()

    drag1_1 = V
    drag2_1 = Times(Length(V), V)

    magnet1_1 = Skew(V)

    # magnet2
    # 1 (op (X (R_hat / R^2) V1) (R_hat / R^2))
    # 2 (* (dp (R_hat / R^2) (R_hat / R^2)) (R_hat / R^2))
    # 3 (op (X (R_hat / R^2) V1) (R_hat / R^2))
    # 4 (op (R_hat / R^2) (X (R_hat / R^2) V1))
    # 5 (op (X (R_hat / R^2) V1) (R_hat / R^2))
    # 6 (op (R_hat / R^2) (X (R_hat / R^2) V1))
    # 7 (X (R_hat / R^2) V1)
    # 8 (skew (* (len (R_hat / R^2)) V1))
    # 9 (op (R_hat / R^2) (X (R_hat / R^2) V1))
    # 10 (skew (* (len (R_hat / R^2)) V1))
    # 11 (op (X (R_hat / R^2) V1) R)
    # 12 (op (X (R_hat / R^2) V1) R)
    # 13 (skew (* (len (R_hat / R^2)) V1))
    # 14 (op (X (R_hat / R^2) V1) R)
    # 15 (* (dp R V1) V1)
    # 16 (X R V1)
    # 17 (op (X R V1) V1)
    # 18 (op (X R V1) V1)
    magnet2_1 = Outer(Cross(AbstractionConst(), V1), AbstractionConst())
    magnet2_2 = Times(Inner(AbstractionConst(), AbstractionConst()), AbstractionConst())
    magnet2_3 = Outer(Cross(AbstractionConst(), V1), AbstractionConst())
    magnet2_4 = Outer(AbstractionConst(), Cross(AbstractionConst(), V1))
    magnet2_5 = magnet2_3
    magnet2_6 = magnet2_4
    magnet2_7 = Cross(AbstractionConst(), V1)
    magnet2_8 = Skew(Times(Length(AbstractionConst()), V1))
    magnet2_9 = magnet2_4
    magnet2_10 = magnet2_8
    magnet2_11 = Outer(Cross(AbstractionConst(), V1), R)
    magnet2_12 = magnet2_11
    magnet2_13 = magnet2_8
    magnet2_14 = magnet2_11
    magnet2_15 = Times(Inner(R, V1), V1)
    magnet2_16 = Cross(R, V1)
    magnet2_17 = Outer(magnet2_16, V1)
    magnet2_18 = magnet2_17

    return [drag3_1, drag3_2,
            falling1, falling2,
            orbit1_1, orbit1_2,
            orbit2_1, orbit2_2, orbit2_3, orbit2_4, orbit2_5, orbit2_6,
            drag1_1,
            drag2_1,
            magnet1_1,
            magnet2_1, magnet2_2, magnet2_3, magnet2_4, magnet2_5, magnet2_6, magnet2_7, magnet2_8,
            magnet2_9, magnet2_10, magnet2_11, magnet2_12, magnet2_13, magnet2_14, magnet2_15,
            magnet2_16, magnet2_17, magnet2_18]


def all_task_solution_expressions():
    # drag3
    # (/ (hat R) (dp R R))
    # (/ R (dp R R))
    # (* (dp V V) V)
    # falling
    # (/ (hat R) (dp R R))
    # orbit
    # (/ (hat R) (dp R R))
    # orbit2
    # (/ (hat R) (dp R R))
    # drag1
    # (* (len V) V)
    # magnet1
    # (skew V)
    # magnet2
    # (skew (/ V1 (dp R R)))
    # (op (X V1 (hat R)) V1)
    R = Vector('R')
    V = Vector('V')
    # V1 = Vector('V1')

    drag3_1 = ScaleInverse(Hat(R), Inner(R, R))
    drag3_2 = ScaleInverse(R, Inner(R, R))
    drag3_3 = Times(Inner(V, V), V)

    falling1 = ScaleInverse(Hat(R), Inner(R, R))

    orbit1 = ScaleInverse(Hat(R), Inner(R, R))

    orbit2_1 = ScaleInverse(Hat(R), Inner(R, R))

    drag1_1 = Times(Length(V), V)

    magnet1_1 = Skew(V)

    magnet2_1 = Skew(ScaleInverse(V, Inner(R, R)))
    # magnet2_2 = Outer(Cross(V1, Hat(R)), V1)

    # return [magnet2_1, magnet2_2]
    return [drag3_1, drag3_2, drag3_3,
            falling1,
            orbit1,
            orbit2_1,
            drag1_1,
            magnet1_1,
            magnet2_1] #, magnet2_2]


def dipole_solution_expressions():
    R = Vector('R')
    V1 = Vector('V1')
    goal1_expr = Skew(ScaleInverse(V1, Inner(R, Scale(Length(R), R))))
    goal2_expr = ScaleInverse(Outer(Cross(V1, R), Hat(R)), Times(Inner(R, R), Inner(R, R)))
    return [goal1_expr, goal2_expr]


def library_learn(expressions):
    programs = [e.pretty_print() for e in expressions]
    print('Library learning on expressions:')
    for p in programs:
        print('\t' + p)
    from stitch_core import compress
    res = compress(programs, iterations=1, max_arity=2)
    print(f'Library learning result: {res.abstractions}')
    return [a.body for a in res.abstractions]


def no_hat_expressions():
    with_hat_expressions = all_task_solution_expressions()
    # replace all instances of ScaleInverse(Hat(R), Inner(R, R)) with (/ R (len (* (dp R R) R)))
    R = Vector('R')
    original = ScaleInverse(Hat(R), Inner(R, R))
    replacement = ScaleInverse(R, Length(Times(Inner(R, R), R)))

    def replace_expr(expr, original, replacement):
        if expr == original:
            return replacement
        else:
            return expr

    return [replace_expr(expr, original, replacement) for expr in with_hat_expressions]


def test_pretty_print_to_expr():
    exprs = all_task_solution_expressions()
    exprs += all_task_solution_expressions_with_abstraction_const()
    exprs += dipole_solution_expressions()
    for expr in exprs:
        pp = expr.pretty_print()
        expr2 = pretty_print_to_expr(pp)
        assert pp == expr2.pretty_print(), f'{pp} != {expr2.pretty_print()}'


def enumerate_bnn_exprs(n=100, n_features=31):
    # simple for now, can get more complicated later
    operators = [
        Divide,
        Times,
        Plus,
        Sin,
    ]

    features = [f'f{i}' for i in range(n_features)]

    def random_input():
        d = {}
        for nm in features:
            d[nm] = random.random()*10
        return d


    variables = [Real(f) for f in features]
    inputs = [random_input() for _ in range(10)]

    expressions = []
    for expression in weighted_bottom_up_generator(cost_bound=4, operators=operators, constants=variables,
                                                  inputs=inputs, cost_dict=None):
        assert expression.return_type == 'real'
        expressions.append(expression)
        if len(expressions) == n:
            break

    return expressions


if __name__ == '__main__':
    es = enumerate_bnn_exprs(n=100, n_features=31)
    for e in es:
        print(e.pretty_print())

