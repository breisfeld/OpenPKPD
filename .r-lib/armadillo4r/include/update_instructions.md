After updating Armadillo version:

1. `armadillo.hpp` includes a custom `r_messages.hpp` in line 28.
2. `armadillo/arma_forward.hpp` omits `std::cout`/`std::cerr` in lines 18/19.
3. `armadillo/config.hpp` calls custom message/error edirections in lines 225/230.
