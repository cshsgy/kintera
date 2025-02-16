#pragma once

// kintera
#include "reaction.hpp"

template <>
struct fmt::formatter<kintera::ArrheniusOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::ArrheniusOptions& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "(A = {}; b = {}; Ea_R = {}; E4_R = {})",
                          p.A(), p.b(), p.Ea_R(), p.E4_R());
  }
};
