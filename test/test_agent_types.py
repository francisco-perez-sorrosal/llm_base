import json
import tempfile
import ast

from pathlib import Path
from llm_foundation import logger
from llm_foundation.agent_types import Example, Persona, Role

def test_persona_to_yaml():
    role1 = Role(name="role1", description="Role 1 Description", agent_system_message="Agent System Message")
    role2 = Role(name="role2", description="Role 2 Description", agent_system_message="Agent System Message again")
    roles_dict= {"role1": role1, "role2": role2}
    persona = Persona(name="Test Persona", roles=roles_dict)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        filename="test_persona.yaml"

        persona.to_yaml_file(Path(temp_dir), filename)
        
        same_persona = Persona.from_yaml_file(str(Path(temp_dir, filename)))

        # Assertions
        assert persona.name == same_persona.name, "Not the same name"
        assert len(persona.roles) == len(same_persona.roles), "Not the same number of roles"
        
        for role_name, role in persona.roles.items():
            same_role = same_persona.roles.get(role_name)
            assert same_role is not None, f"Role {role_name} not found in deserialized persona"
            assert role.name == same_role.name, f"Role name mismatch for {role_name}"
            assert role.description == same_role.description, f"Role description mismatch for {role_name}"
            assert role.agent_system_message == same_role.agent_system_message, f"Agent system message mismatch for {role_name}"

def test_persona_from_yaml():
    yaml_content = """
    name: Test Persona
    roles:
        role1:
            name: role1
            description: Role 1 Description
            agent_system_message: Agent System Message
            tasks:
                - name: task1
                  description: This is task 1
                  expected_output: This is expected output for task 1
                - name: task2
                  description: This is task 2
                  expected_output: This is expected output for task 2
        role2:
            name: role2
            description: Role 2 Description
            agent_system_message: Agent System Message again
            examples:
                - header: example 1
                  format: text
                  content: This is an example of example 1
                - header: example 2
                  format: json
                  content: |
                    {
                        'header': 'example 2', 
                        'format': 'json', 
                        'content': 'This is an example of example 2'
                    }
            
    """
    
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_persona.yaml"
        with open(file_path, 'w') as file:
            file.write(yaml_content)
        
        persona = Persona.from_yaml_file(str(file_path))

        # Assertions
        assert persona.name == "Test Persona", "Persona name mismatch"
        assert len(persona.roles) == 2, "Number of roles mismatch"
        
        role1 = persona.roles.get("role1")
        assert role1 is not None, "Role role1 not found"
        assert role1.name == "role1", "Role1 name mismatch"
        assert role1.description == "Role 1 Description", "Role1 description mismatch"
        assert role1.agent_system_message == "Agent System Message", "Role1 agent system message mismatch"
        assert len(role1.tasks) == 2, "Number of tasks in role1 mismatch"
        
        task1 = role1.tasks[0]
        assert task1.name == "task1", "Task1 name mismatch"
        assert task1.description == "This is task 1", "Task1 description mismatch"
        assert task1.expected_output == "This is expected output for task 1", "Task1 expected output mismatch"
        
        task2 = role1.tasks[1]
        assert task2.name == "task2", "Task2 name mismatch"
        assert task2.description == "This is task 2", "Task2 description mismatch"
        assert task2.expected_output == "This is expected output for task 2", "Task2 expected output mismatch"
                
        role2 = persona.roles.get("role2")
        assert role2 is not None, "Role role2 not found"
        assert role2.name == "role2", "Role2 name mismatch"
        assert role2.description == "Role 2 Description", "Role2 description mismatch"
        assert role2.agent_system_message == "Agent System Message again", "Role2 agent system message mismatch"
        assert len(role2.tasks) == 0, "Number of tasks in role2 mismatch"

        assert len(role2.examples) == 2, "Number of examples in role2 mismatch"        
        example1 = role2.examples[0]
        assert example1.header == "example 1", "Example1 header mismatch"
        assert example1.format == "text", "Example1 format mismatch"
        assert example1.content == "This is an example of example 1", "Example1 content mismatch"
        
        example2 = role2.examples[1]
        assert example2.header == "example 2", "Example2 header mismatch"
        assert example2.format == "json", "Example2 format mismatch"
        if example2.format == "json":
            example2_json = ast.literal_eval(example2.content)
            assert example2_json['header'] == "example 2", "Example2 JSON header mismatch"
            assert example2_json['format'] == "json", "Example2 JSON format mismatch"
            assert example2_json['content'] == "This is an example of example 2", "Example2 JSON content mismatch"
        
        # from rich.pretty import pprint
        # from io import StringIO
        # from contextlib import redirect_stdout
        # str_io = StringIO()
        # with redirect_stdout(str_io):
        #     pprint(role1)
        # output = str_io.getvalue()
        # logger.warning(output)


def test_get_examples_as_str_from_role():
    examples = [
        Example(header="example 1", format="json", content='{"key": "value"}'),
        Example(header="example 2", format="json", content='{"another_key": "another_value"}')
    ]
    role = Role(
        name="role_with_examples",
        description="Role with examples",
        agent_system_message="Agent System Message",
        examples=examples
    )
    
    examples_str = role.get_examples_as_str()
    
    logger.info(examples_str)
    
    expected_str = "example 1" + "\n\n" + json.dumps({"key": "value"}, indent=4) + "\n\n" + "example 2" + "\n\n" + json.dumps({"another_key": "another_value"}, indent=4) + "\n\n"
    
    assert examples_str == expected_str, "Examples string mismatch"
    