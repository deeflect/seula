const params = new URLSearchParams(location.search);
if (params.has("lab")) {
  void import("./lab");
} else if (params.has("invoice")) {
  void import("./invoice");
} else {
  void import("./live");
}
