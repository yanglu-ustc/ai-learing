$$
\begin{cases}
\det(I_n-BA)=\det\left(\begin{matrix}I_m & A\\B & I_n\end{matrix}\right)=\det(I_m-AB) \implies \det(A-\bm{a}\bm{a}^T)=(1-\bm{a}^TA^{-1}\bm{a})\det(A) \\
\\
m+rank(I_n-BA)=rank\left(\begin{matrix}I_m & A\\B & I_n\end{matrix}\right)=n+rank(I_m-AB) \\
\\
(I_n-BA)^{-1}=I_n+B(I_m-AB)^{-1}A \\
\\
\lambda^n\det(\lambda I_m-AB)=\lambda^m\det(\lambda I_n-BA)
\end{cases}
$$

$$
(A+\bm{a}\bm{b}^T)^{-1}=A^{-1}-\frac{A^{-1}\bm{a}\bm{b}^TA^{-1}}{1+\bm{b}^TA^{-1}\bm{a}}
$$

$$
\begin{cases}
rank\left(\begin{matrix}A & B\end{matrix}\right)&\leq rank(A)+rank(B) \\
\\
rank\left(\begin{matrix}A & B\end{matrix}\right)&\geq\max(rank(A),rank(B),rank(A+B)) \\
\end{cases}
$$

$$
\begin{cases}
rank(AB)\leq\min(rank(A),rank(B)) \\
\\
rank(AB)\geq rank(A)+rank(B)-n \\
\\
rank(ABC)+rank(B)\geq rank(AB)+rank(BC)
\end{cases}
$$
