data:extend({
  {
    type = "string-setting",
    name = "factorion-default-item",
    setting_type = "runtime-global",
    default_value = "iron-plate",
    order = "a",
  },
  {
    type = "string-setting",
    name = "factorion-checkpoint",
    setting_type = "runtime-global",
    default_value = "mvbipbr0",
    order = "b",
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
  {
    type = "double-setting",
    name = "factorion-temperature",
    setting_type = "runtime-global",
    default_value = 1,
    minimum_value = 0,
    maximum_value = 2,
    order = "d",
  },
})
