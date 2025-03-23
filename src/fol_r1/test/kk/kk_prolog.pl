% Atomic formula
eval_atomic(Predicate) :-
    call(Predicate).

% Negation
eval(not(F)) :-
    \+ eval(F).

% Conjunction
eval(and(F1, F2)) :-
    eval(F1),
    eval(F2).

% Disjunction
eval(or(F1, F2)) :-
    (eval(F1); eval(F2)).

% Implication (A → B ≡ ¬A ∨ B)
eval(implies(A, B)) :-
    (eval(not(A)); eval(B)).

% Equivalence
eval(equiv(A, B)) :-
    (eval(A), eval(B));
    (eval(not(A)), eval(not(B))).

% Existential quantifier
eval(exists(Var, F)) :-
    % instantiate Var to some value in the universe
    % Here, just trying all humans as the domain
    human(Var),
    eval(F).

% Universal quantifier
eval(forall(Var, F)) :-
    % succeeds if F is true for all humans
    \+ (human(Var), \+ eval(F)).

% Base case for atomic formulas
eval(F) :-
    F =.. [Pred | Args],
    call(F).