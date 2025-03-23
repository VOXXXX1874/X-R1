% ----------------------------
% Propositional Logic in Prolog
% ----------------------------

% Define propositional operators as nested terms:
% atom -> just an atom like 'p', 'q', etc.
% not(F) -> negation
% and(F1, F2) -> conjunction
% or(F1, F2) -> disjunction
% implies(F1, F2) -> implication
% equiv(F1, F2) -> biconditional

% ----------------------------
% Example truth assignment (interpretation)
% ----------------------------
value(p, true).
value(q, false).
value(r, true).
value(s, false).

% ----------------------------
% Evaluator
% ----------------------------

% Base case: propositional atom
eval(Atom, Value) :-
    atom(Atom),
    value(Atom, Value).

% Negation
eval(not(F), true) :-
    eval(F, false).
eval(not(F), false) :-
    eval(F, true).

% Conjunction
eval(and(F1, F2), true) :-
    eval(F1, true),
    eval(F2, true).
eval(and(F1, F2), false) :-
    (eval(F1, false); eval(F2, false)).

% Disjunction
eval(or(F1, F2), true) :-
    (eval(F1, true); eval(F2, true)).
eval(or(F1, F2), false) :-
    eval(F1, false),
    eval(F2, false).

% Implication
eval(implies(F1, F2), true) :-
    (eval(F1, false); eval(F2, true)).
eval(implies(F1, F2), false) :-
    eval(F1, true),
    eval(F2, false).

% Biconditional (equivalence)
eval(equiv(F1, F2), true) :-
    eval(F1, V),
    eval(F2, V).
eval(equiv(F1, F2), false) :-
    eval(F1, V1),
    eval(F2, V2),
    V1 \= V2.

% ----------------------------
% Example formulas and queries
% ----------------------------

% Example formulas:
% (p ∧ q) → r
formula1(implies(and(p, q), r)).

% ¬p ∨ s
formula2(or(not(p), s)).

% (p ↔ r)
formula3(equiv(p, r)).

% ----------------------------
% Example queries:
% ----------------------------
% ?- formula1(F), eval(F, V).
% ?- formula2(F), eval(F, V).
% ?- formula3(F), eval(F, V).