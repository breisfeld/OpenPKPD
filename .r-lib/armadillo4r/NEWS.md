# armadillo4r 0.7.0

* Using Armadillo 15.2.1 (Medium Roast Deluxe) thanks to @conradsnicta update notification.
* Improve complex vectors/matrices copying performance by using `memcpy` instead of element-wise copying.
* Explicit testing with Clang and GCC compiler using C++ 14, 17, 20, and 23 standards (Armadillo 15.2.1 
  requires at least C++14)
  
# armadillo4r 0.6.0

* cpp11armadillo was renamed to armadillo4r to reflect its purpose better
* supports copying complex numbers between R and C++
