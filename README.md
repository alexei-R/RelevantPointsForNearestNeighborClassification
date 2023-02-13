# Relevant Points for Nearest Neighbor Classification
This repository contains implementations of the algorithms of Eppstein and Flores-Velazco for finding relevant points for nearest neighbor classification.

D. Eppstein. Finding relevant points for nearest-neighbor classification. In Symposium on
Simplicity in Algorithms (SOSA), pages 68–78, 2022.

A. Flores-Velazco. Improved search of relevant points for nearest-neighbor classification. In
S. Chechik, G. Navarro, E. Rotenberg, and G. Herman, editors, 30th Annual European Symposium on Algorithms (ESA 2022), volume 244 of Leibniz International Proceedings in Informatics (LIPIcs), pages 54:1–54:10, Dagstuhl, Germany, 2022. Schloss Dagstuhl – Leibniz-Zentrum
für Informatik.

An important subroutine is an algorithm for finding the extreme points in a set proposed by Clarkson (slightly modified here).

K. Clarkson. More output-sensitive geometric algorithms. In Proceedings 35th Annual Symposium on Foundations of Computer Science, pages 695–702, 1994.

It also relies on the implementation of Seidel's algorithm for solving linear programs originating from this repository https://github.com/ZJU-FAST-Lab/SDLP belonging to Zhepei Wang.

R. Seidel. Small-dimensional linear programming and convex hulls made easy. Discrete &
Computational Geometry, 6(3):423–434, Sep 1991.

Eppstein's algorithm also required the implementation of the well-known Jarník's algorithm for finding the minimum spanning tree of a graph, for the special case of the complete Euclidian graph.
