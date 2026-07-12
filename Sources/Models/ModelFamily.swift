/// Declarative model family categories shared by graph and runtime routing.
public enum ModelFamily: String, CaseIterable, Sendable {
    case transformer
    case gemma3Text
    case gemma4
    case qwen35
    case lfm2
    case cohere
}
