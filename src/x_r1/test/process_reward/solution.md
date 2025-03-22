We count the number of times $\frac{1}{n^3}$ appears in the sum
\[\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3},\]where $n$ is a fixed positive integer. (In other words, we are conditioning the sum on $j + k$.) We get a term of $\frac{1}{n^3}$ each time $j + k = n.$ The pairs $(j,k)$ that work are $(1,n - 1),$ $(2,n - 2),$ $\dots,$ $(n - 1,1),$ for a total of $n - 1$ pairs. Therefore,
\begin{align*}
\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3} &= \sum_{n = 1}^\infty \frac{n - 1}{n^3} \\
&= \sum_{n = 1}^\infty \left( \frac{n}{n^3} - \frac{1}{n^3} \right) \\
&= \sum_{n = 1}^\infty \left( \frac{1}{n^2} - \frac{1}{n^3} \right) \\
&= \sum_{n = 1}^\infty \frac{1}{n^2} - \sum_{n = 1}^\infty \frac{1}{n^3} \\
&= \boxed{p - q}.
\end{align*}