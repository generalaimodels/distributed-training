export function formatLongDate(value: string | null): string {
  if (!value) {
    return "Living note";
  }

  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "long",
  }).format(new Date(value));
}

export function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}
