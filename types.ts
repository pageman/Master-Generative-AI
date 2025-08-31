
export interface Stat {
  number: string;
  label: string;
}

export interface Module {
  icon: string;
  title: string;
  duration: string;
  description: string;
  skills: string[];
  difficulty: string;
  difficultyClass: string;
}

export interface Resource {
  type: string;
  title: string;
  description: string;
}

export interface Project {
    title: string;
    description: string;
    codeId: string;
    codeLang: string;
    code: string;
}

export interface AccordionItemData {
  icon: string;
  title: string;
  learningObjectives: string;
  keyTopics: string[];
  projects: Project[];
}
