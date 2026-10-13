-- Shared by settings.lua (registers them) and control.lua (falls back to
-- these defaults when the running Factorio predates a setting).
return {
  {
    type = "string-setting",
    name = "factorion-default-source-item",
    setting_type = "runtime-global",
    default_value = "iron-plate",
    order = "a",
  },
  {
    type = "double-setting",
    name = "factorion-sink-smoothing",
    setting_type = "runtime-global",
    default_value = 2,
    minimum_value = 0.05,
    maximum_value = 30,
    order = "a3",
  },
  {
    type = "string-setting",
    name = "factorion-default-sink-item",
    setting_type = "runtime-global",
    default_value = "iron-gear-wheel",
    order = "a2",
  },
  {
    type = "string-setting",
    name = "factorion-placement-mode",
    setting_type = "runtime-global",
    default_value = "ghost",
    allowed_values = { "ghost", "build" },
    order = "b2",
  },
  {
    type = "double-setting",
    name = "factorion-placement-delay",
    setting_type = "runtime-global",
    default_value = 0.01,
    minimum_value = 0,
    maximum_value = 1,
    order = "c",
  },
}
